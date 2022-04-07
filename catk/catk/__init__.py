""" Package catk Correspondence Analysis Toolkit: main module.
    Import submodules into the main one.
"""


from . import data
from . import tools
from .ca import CA # type: ignore
from .config import SUM_SYMBOL

__version__ = '0.0.3'

print(f"'{__file__}' loaded")
