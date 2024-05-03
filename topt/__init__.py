import importlib.metadata

from . import converter, optimizer

__version__ = importlib.metadata.version("topt")

__all__ = [
    "__version__",
    "converter",
    "optimizer",
]
