""" Package catk Correspondence Analysis Toolkit: main module.
    Import submodules into the main one.
"""


from . import data
from . import tools
from .ca import CA # type: ignore


print(f"'{__file__}' loaded")
