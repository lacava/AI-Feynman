"""AI Feynman
"""

from .sklearn import AIFeynmanRegressor
from .S_run_aifeynman import run_aifeynman
from .get_demos import get_demos

__title__ = "aifeynman"
__version__ = "2.0.7.6"
__license__ = "MIT"
__copyright__ = "Copyright 2020 Silviu-Marian Udrescu"

__all__ = ["run_aifeynman", "AIFeynmanRegressor"]
