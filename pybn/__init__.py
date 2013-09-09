"""Python Bayesian Networks

"""

__version__ = '1.1'


try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy does not seem to be installed. Please see the user guide.')

from network import *
from operations import *