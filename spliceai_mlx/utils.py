"""
SpliceAI utils — pure MLX implementation.

Public API (identical to original utils.py):
    Annotator
    get_delta_scores
    one_hot_encode
    normalise_chrom
"""

import os
import logging

import numpy as np
import pandas as pd
import mlx.core as mx
from pyfaidx import Fasta


# ---------------------------------------------------------------------------
# Package-relative resource helper
# ---------------------------------------------------------------------------

def _resource_path(relative_path: str) -> str:
    """Return the absolute path to a file inside the spliceai package directory."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, relative_path)


# ---------------------------------------------------------------------------
# MLX forward pass
# ---------------------------------------------------------------------------
# Architecture (derived from model.summary() and JSON topology of spliceai*.h5):
#
#   Input: (1, L, 4)  — one-hot encoded DNA, float32
#
#   Stem:
#     conv1d_1  k=1  d=1  → BN1 → ReLU            [c1 = pre-BN output saved for skip]
#
#   Group 1  (4 res-blocks, k=11, d=1):
#     block 1: act1 → conv3→BN2→ReLU→conv4  + c1   = add_1
#     block 2: BN3→ReLU(add_1) → conv5→BN4→ReLU→conv6  + add_1  = add_2
#     block 3: BN5→ReLU(add_2) → conv7→BN6→ReLU→conv8  + add_2  = add_3
#     block 4: BN7→ReLU(add_3) → conv9→BN8→ReLU→conv10 + add_3  = add_4
#
#   Group 2  (4 res-blocks, k=11, d=4):   → add_9
#   Group 3  (4 res-blocks, k=21, d=10):  → add_14
#   Group 4  (4 res-blocks, k=41, d=25):  → add_18
#     Extra branch: BN31→ReLU(add_18)→conv36→BN32→ReLU→conv37 + add_18 = add_19
#
#   Multi-scale skip merge (all 1×1 projections):
#     add_20 = conv2(c1) + conv11(add4) + conv20(add9) + conv29(add14) + conv38(add19)
#
#   Crop 5000 positions from each end  → (1, L-10000, 32)
#   Output: conv1d_39 k=1 → softmax   → (1, L-10000, 3)
# ---------------------------------------------------------------------------

def _conv1d(
    x: mx.array,
    kernel: np.ndarray,
    bias: np.ndarray,
    padding: int = 0,
    dilation: int = 1,
) -> mx.array:
    """Wrap MLX conv1d.  Weights stored as (k, C_in, C_out); MLX expects (C_out, k, C_in)."""
    w = mx.transpose(mx.array(kernel), (2, 0, 1))
    b = mx.array(bias)
    return mx.conv1d(x, w, padding=padding, dilation=dilation) + b


def _bn(x: mx.array, weights: dict, name: str) -> mx.array:
    """Batch normalisation in inference mode (eps=1e-3, matching Keras default)."""
    g = mx.array(weights[f'{name}/gamma'])
    b = mx.array(weights[f'{name}/beta'])
    m = mx.array(weights[f'{name}/moving_mean'])
    v = mx.array(weights[f'{name}/moving_var'])
    return g * (x - m) / mx.sqrt(v + 1e-3) + b


def _spliceai_forward(x_np: np.ndarray, weights: dict) -> np.ndarray:
    """
    Single SpliceAI forward pass using MLX.

    Parameters
    ----------
    x_np : np.ndarray, shape (1, L, 4)
    weights : dict  key → np.ndarray  (loaded from .npz)

    Returns
    -------
    np.ndarray, shape (1, L-10000, 3)  splice-site probabilities
    """
    x = mx.array(x_np.astype(np.float32))

    def cw(name: str, inp: mx.array, k: int, d: int) -> mx.array:
        pad = (k - 1) * d // 2
        return _conv1d(inp, weights[f'{name}/kernel'], weights[f'{name}/bias'],
                       padding=pad, dilation=d)

    def br(name: str, inp: mx.array) -> mx.array:
        return mx.maximum(_bn(inp, weights, name), 0)

    # ---- Stem ----
    c1   = cw('conv1d_1', x, 1, 1)
    act1 = mx.maximum(_bn(c1, weights, 'batch_normalization_1'), 0)

    # ---- Group 1: k=11, d=1 ----
    h    = br('batch_normalization_2', cw('conv1d_3', act1, 11, 1))
    add1 = cw('conv1d_4', h, 11, 1) + c1

    h    = br('batch_normalization_4', cw('conv1d_5', br('batch_normalization_3', add1), 11, 1))
    add2 = cw('conv1d_6', h, 11, 1) + add1

    h    = br('batch_normalization_6', cw('conv1d_7', br('batch_normalization_5', add2), 11, 1))
    add3 = cw('conv1d_8', h, 11, 1) + add2

    h    = br('batch_normalization_8', cw('conv1d_9', br('batch_normalization_7', add3), 11, 1))
    add4 = cw('conv1d_10', h, 11, 1) + add3

    # ---- Group 2: k=11, d=4 ----
    h    = br('batch_normalization_10', cw('conv1d_12', br('batch_normalization_9',  add4), 11, 4))
    add6 = cw('conv1d_13', h, 11, 4) + add4

    h    = br('batch_normalization_12', cw('conv1d_14', br('batch_normalization_11', add6), 11, 4))
    add7 = cw('conv1d_15', h, 11, 4) + add6

    h    = br('batch_normalization_14', cw('conv1d_16', br('batch_normalization_13', add7), 11, 4))
    add8 = cw('conv1d_17', h, 11, 4) + add7

    h    = br('batch_normalization_16', cw('conv1d_18', br('batch_normalization_15', add8), 11, 4))
    add9 = cw('conv1d_19', h, 11, 4) + add8

    # ---- Group 3: k=21, d=10 ----
    h     = br('batch_normalization_18', cw('conv1d_21', br('batch_normalization_17', add9),  21, 10))
    add11 = cw('conv1d_22', h, 21, 10) + add9

    h     = br('batch_normalization_20', cw('conv1d_23', br('batch_normalization_19', add11), 21, 10))
    add12 = cw('conv1d_24', h, 21, 10) + add11

    h     = br('batch_normalization_22', cw('conv1d_25', br('batch_normalization_21', add12), 21, 10))
    add13 = cw('conv1d_26', h, 21, 10) + add12

    h     = br('batch_normalization_24', cw('conv1d_27', br('batch_normalization_23', add13), 21, 10))
    add14 = cw('conv1d_28', h, 21, 10) + add13

    # ---- Group 4: k=41, d=25 ----
    h     = br('batch_normalization_26', cw('conv1d_30', br('batch_normalization_25', add14), 41, 25))
    add16 = cw('conv1d_31', h, 41, 25) + add14

    h     = br('batch_normalization_28', cw('conv1d_32', br('batch_normalization_27', add16), 41, 25))
    add17 = cw('conv1d_33', h, 41, 25) + add16

    h     = br('batch_normalization_30', cw('conv1d_34', br('batch_normalization_29', add17), 41, 25))
    add18 = cw('conv1d_35', h, 41, 25) + add17

    # Extra branch (block 4 + projection)
    h     = cw('conv1d_36', br('batch_normalization_31', add18), 41, 25)
    act32 = mx.maximum(_bn(h, weights, 'batch_normalization_32'), 0)
    add19 = cw('conv1d_37', act32, 41, 25) + add18

    # ---- Multi-scale skip merge (all 1×1 projections) ----
    c2  = _conv1d(c1,    weights['conv1d_2/kernel'],  weights['conv1d_2/bias'])
    c11 = _conv1d(add4,  weights['conv1d_11/kernel'], weights['conv1d_11/bias'])
    c20 = _conv1d(add9,  weights['conv1d_20/kernel'], weights['conv1d_20/bias'])
    c29 = _conv1d(add14, weights['conv1d_29/kernel'], weights['conv1d_29/bias'])
    c38 = _conv1d(add19, weights['conv1d_38/kernel'], weights['conv1d_38/bias'])

    add20 = (c2 + c11) + c20 + c29 + c38

    # ---- Crop 5000 positions from each end ----
    crop = add20[:, 5000:-5000, :]

    # ---- Final 1×1 conv + softmax ----
    out = _conv1d(crop, weights['conv1d_39/kernel'], weights['conv1d_39/bias'])
    out = mx.softmax(out, axis=-1)

    mx.eval(out)
    return np.array(out)


class SpliceAIModel:
    """Wraps a single SpliceAI model loaded from a .npz weights file."""

    def __init__(self, weights_path: str) -> None:
        data = np.load(weights_path)
        self.weights = {k: data[k] for k in data.files}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray, shape (1, L, 4)

        Returns
        -------
        np.ndarray, shape (1, L-10000, 3)
        """
        return _spliceai_forward(x, self.weights)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Annotator:

    def __init__(self, ref_fasta: str, annotations: str) -> None:

        if annotations == 'grch37':
            annotations = _resource_path('annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = _resource_path('annotations/grch38.txt')

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes      = df['#NAME'].to_numpy()
            self.chroms     = df['CHROM'].to_numpy()
            self.strands    = df['STRAND'].to_numpy()
            self.tx_starts  = df['TX_START'].to_numpy() + 1
            self.tx_ends    = df['TX_END'].to_numpy()
            self.exon_starts = [
                np.asarray([int(i) for i in c.split(',') if i]) + 1
                for c in df['EXON_START'].to_numpy()
            ]
            self.exon_ends = [
                np.asarray([int(i) for i in c.split(',') if i])
                for c in df['EXON_END'].to_numpy()
            ]
        except OSError as e:
            logging.error(str(e))
            raise
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file %s not formatted properly: %s', annotations, e)
            raise

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except OSError as e:
            logging.error(str(e))
            raise

        self.models = [
            SpliceAIModel(_resource_path(f'models/spliceai{i}.npz'))
            for i in range(1, 6)
        ]

    def get_name_and_strand(self, chrom: str, pos: int):
        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(
            np.nonzero(self.chroms == chrom)[0],
            np.intersect1d(
                np.nonzero(self.tx_starts <= pos)[0],
                np.nonzero(pos <= self.tx_ends)[0],
            ),
        )
        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        return [], [], []

    def get_pos_data(self, idx, pos: int):
        dist_tx_start  = self.tx_starts[idx] - pos
        dist_tx_end    = self.tx_ends[idx] - pos
        dist_exon_bdry = min(
            np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos,
            key=abs,
        )
        return (dist_tx_start, dist_tx_end, dist_exon_bdry)


def one_hot_encode(seq: str) -> np.ndarray:
    """Map a DNA string to a (L, 4) float32 one-hot array."""
    table = np.array(
        [[0, 0, 0, 0],   # unknown / N
         [1, 0, 0, 0],   # A
         [0, 1, 0, 0],   # C
         [0, 0, 1, 0],   # G
         [0, 0, 0, 1]],  # T
        dtype=np.float32,
    )
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    return table[np.frombuffer(seq.encode('latin-1'), dtype=np.uint8) % 5]


def normalise_chrom(source: str, target: str) -> str:
    """Ensure `source` chromosome name has/lacks the 'chr' prefix to match `target`."""
    src_has = source.startswith('chr')
    tgt_has = target.startswith('chr')
    if src_has and not tgt_has:
        return source[3:]
    if not src_has and tgt_has:
        return 'chr' + source
    return source


def get_delta_scores(record, ann: Annotator, dist_var: int, mask: int) -> list[str]:
    """
    Compute SpliceAI delta scores for a single VCF record.

    Parameters
    ----------
    record    : object with .chrom, .pos, .ref, .alts attributes
    ann       : Annotator instance
    dist_var  : maximum distance between variant and splice site to report
    mask      : 1 = mask annotated sites, 0 = report all

    Returns
    -------
    list of score strings, one per alt × gene combination
    """
    cov = 2 * dist_var + 1
    wid = 10000 + cov
    delta_scores: list[str] = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): %s', record)
        return delta_scores

    genes, strands, idxs = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): %s', record)
        return delta_scores

    if seq[wid // 2 : wid // 2 + len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref mismatch): %s', record)
        return delta_scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): %s', record)
        return delta_scores

    if len(record.ref) > 2 * dist_var:
        logging.warning('Skipping record (ref too long): %s', record)
        return delta_scores

    for j in range(len(record.alts)):
        alt = str(record.alts[j])

        if any(c in alt for c in ('.', '-', '*', '<', '>')):
            continue

        for i in range(len(idxs)):
            if len(record.ref) > 1 and len(alt) > 1:
                delta_scores.append(f'{alt}|{genes[i]}|.|.|.|.|.|.|.|.')
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
            ref_len  = len(record.ref)
            alt_len  = len(alt)
            del_len  = max(ref_len - alt_len, 0)

            x_ref = 'N' * pad_size[0] + seq[pad_size[0] : wid - pad_size[1]] + 'N' * pad_size[1]
            x_alt = x_ref[: wid // 2] + alt + x_ref[wid // 2 + ref_len :]

            x_ref = one_hot_encode(x_ref)[np.newaxis, :]
            x_alt = one_hot_encode(x_alt)[np.newaxis, :]

            if strands[i] == '-':
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(5)], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(5)], axis=0)

            if strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2 + alt_len],
                    np.zeros((1, del_len, 3), dtype=np.float32),
                    y_alt[:, cov // 2 + alt_len :],
                ], axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2],
                    np.max(y_alt[:, cov // 2 : cov // 2 + alt_len], axis=1)[:, np.newaxis, :],
                    y_alt[:, cov // 2 + alt_len :],
                ], axis=1)

            y = np.concatenate([y_ref, y_alt])

            idx_pa = int((y[1, :, 1] - y[0, :, 1]).argmax())
            idx_na = int((y[0, :, 1] - y[1, :, 1]).argmax())
            idx_pd = int((y[1, :, 2] - y[0, :, 2]).argmax())
            idx_nd = int((y[0, :, 2] - y[1, :, 2]).argmax())

            mask_pa = bool((idx_pa - cov // 2 == dist_ann[2]) and mask)
            mask_na = bool((idx_na - cov // 2 != dist_ann[2]) and mask)
            mask_pd = bool((idx_pd - cov // 2 == dist_ann[2]) and mask)
            mask_nd = bool((idx_nd - cov // 2 != dist_ann[2]) and mask)

            delta_scores.append(
                '{alt}|{gene}|{ds_ag:.2f}|{ds_al:.2f}|{ds_dg:.2f}|{ds_dl:.2f}'
                '|{dp_ag}|{dp_al}|{dp_dg}|{dp_dl}'.format(
                    alt    = alt,
                    gene   = genes[i],
                    ds_ag  = (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),
                    ds_al  = (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),
                    ds_dg  = (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),
                    ds_dl  = (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),
                    dp_ag  = idx_pa - cov // 2,
                    dp_al  = idx_na - cov // 2,
                    dp_dg  = idx_pd - cov // 2,
                    dp_dl  = idx_nd - cov // 2,
                )
            )

    return delta_scores
