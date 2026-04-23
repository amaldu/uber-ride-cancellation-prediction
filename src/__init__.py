# Import directly from the utility modules
from .eda_utils.cleaning_utils import to_snake_case, clean
from .eda_utils.univar_utils import get_stats, get_outliers, get_nans

__all__ = [
    'to_snake_case',
    'clean',
    'get_stats',
    'get_outliers', 
    'get_nans',
]
