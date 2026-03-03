"""src — shared utilities for the Uber ride cancellation project."""

from .eda_utils import get_stats, cramers_v, get_outliers, get_nans
from .evaluation import (
    METRIC_TARGETS,
    evaluate_model,
    find_optimal_threshold,
    plot_evaluation,
)
from .feature_engineering import (
    create_target_encoding,
    create_temporal_features,
    create_vehicle_encoding,
    create_vtat_features,
    group_infrequent_locations,
)
