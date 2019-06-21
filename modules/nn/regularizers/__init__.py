"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from modules.nn.regularizers.regularizer import Regularizer
from modules.nn.regularizers.regularizers import L1Regularizer
from modules.nn.regularizers.regularizers import L2Regularizer
from modules.nn.regularizers.regularizer_applicator import RegularizerApplicator