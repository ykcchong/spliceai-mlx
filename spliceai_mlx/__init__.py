import signal
from importlib.metadata import version, PackageNotFoundError

signal.signal(signal.SIGINT, lambda x, y: exit(0))

name = 'spliceai-mlx'
try:
    __version__ = version(name)
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'
