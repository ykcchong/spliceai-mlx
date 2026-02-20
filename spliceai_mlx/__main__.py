import sys
import argparse
import logging

import pysam

from spliceai_mlx.utils import Annotator, get_delta_scores

_VERSION = '1.3.1'


def get_options():
    parser = argparse.ArgumentParser(description=f'SpliceAI-MLX v{_VERSION} (MLX)')
    parser.add_argument('-I', metavar='input',  nargs='?', default=sys.stdin.buffer,
                        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', metavar='output', nargs='?', default=sys.stdout.buffer,
                        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-D', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-M', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    return parser.parse_args()


def main():
    args = get_options()

    if None in (args.I, args.O, args.D, args.M):
        logging.error(
            'Usage: spliceai-mlx [-h] [-I [input]] [-O [output]] -R reference -A annotation '
            '[-D [distance]] [-M [mask]]'
        )
        raise SystemExit(1)

    try:
        vcf = pysam.VariantFile(args.I)
    except (OSError, ValueError) as e:
        logging.error(str(e))
        raise SystemExit(1)

    header = vcf.header
    header.add_line(
        f'##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv{_VERSION} variant '
        'annotation. These include delta scores (DS) and delta positions (DP) for '
        'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
        'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">'
    )

    try:
        output = pysam.VariantFile(args.O, mode='w', header=header)
    except (OSError, ValueError) as e:
        logging.error(str(e))
        raise SystemExit(1)

    ann = Annotator(args.R, args.A)

    for record in vcf:
        scores = get_delta_scores(record, ann, args.D, args.M)
        if scores:
            record.info['SpliceAI'] = scores
        output.write(record)

    vcf.close()
    output.close()


if __name__ == '__main__':
    main()
