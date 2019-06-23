"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from core.nn.regularizers.regularizer import Regularizer
from core.nn.regularizers.regularizers import L1Regularizer
from core.nn.regularizers.regularizers import L2Regularizer
from core.nn.regularizers.regularizer_applicator import RegularizerApplicator