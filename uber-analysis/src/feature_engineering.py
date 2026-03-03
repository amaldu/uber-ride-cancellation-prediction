"""Feature engineering transformations for the Uber cancellation dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_temporal_features(input_df):
    """Derive time-based features from date + time columns."""
    result_df = input_df.copy()

    result_df['datetime'] = pd.to_datetime(
        result_df['date'] + " " + result_df['time'], format="%Y-%m-%d %H:%M:%S"
    )

    result_df['hour'] = result_df['datetime'].dt.hour
    result_df['dayofweek'] = result_df['datetime'].dt.dayofweek
    result_df['month'] = result_df['datetime'].dt.month
    result_df['day'] = result_df['datetime'].dt.day

    result_df['is_weekend'] = (result_df['dayofweek'] >= 5).astype(int)
    result_df['is_morning_rush'] = ((result_df['hour'] >= 7) & (result_df['hour'] <= 10)).astype(int)
    result_df['is_evening_rush'] = ((result_df['hour'] >= 17) & (result_df['hour'] <= 21)).astype(int)
    result_df['is_peak_hour'] = (result_df['is_morning_rush'] | result_df['is_evening_rush']).astype(int)
    result_df['is_late_night'] = ((result_df['hour'] >= 23) | (result_df['hour'] <= 5)).astype(int)

    result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
    result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)

    result_df['dow_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
    result_df['dow_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)

    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)

    return result_df


def create_vtat_features(input_df, train_medians=None):
    """Create VTAT-derived features with optional pre-computed medians for val/test."""
    result_df = input_df.copy()

    if train_medians is None:
        train_medians = result_df.groupby('vehicle_type', observed=True)['avg_vtat'].median().to_dict()
        return_medians = True
    else:
        return_medians = False

    result_df['avg_vtat_imputed'] = result_df.apply(
        lambda row: row['avg_vtat'] if pd.notna(row['avg_vtat'])
        else train_medians.get(row['vehicle_type'], result_df['avg_vtat'].median()),
        axis=1,
    )

    bins = [0, 5, 10, 15, 20]
    labels = [0, 1, 2, 3]
    result_df['vtat_bucket'] = pd.cut(result_df['avg_vtat_imputed'], bins=bins, labels=labels).astype(float)

    result_df['is_high_vtat'] = (result_df['avg_vtat_imputed'] >= 15).astype(int)

    if return_medians:
        return result_df, train_medians
    return result_df


def group_infrequent_locations(input_df, column, top_n=10, top_locations=None):
    """Keep top-N locations by frequency; bucket the rest into 'Other'."""
    result_df = input_df.copy()
    new_col_name = f"{column}_grouped"

    if top_locations is None:
        top_locations = result_df[column].value_counts().head(top_n).index.tolist()
        return_top = True
    else:
        return_top = False

    result_df[new_col_name] = result_df[column].apply(
        lambda x: x if x in top_locations else 'Other'
    )

    if return_top:
        return result_df, top_locations
    return result_df


def create_target_encoding(input_df, column, target_col='is_cancelled', train_means=None, smoothing=10):
    """Smoothed target encoding; learns from training data or applies pre-computed means."""
    if train_means is None:
        global_mean = input_df[target_col].mean()
        agg = input_df.groupby(column, observed=True)[target_col].agg(['mean', 'count'])

        smoothed_means = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        means_dict = smoothed_means.to_dict()
        means_dict['__global__'] = global_mean

        encoded = input_df[column].map(means_dict).fillna(global_mean)
        return encoded, means_dict
    else:
        global_mean = train_means.get('__global__', 0.32)
        encoded = input_df[column].map(train_means).fillna(global_mean)
        return encoded


def create_vehicle_encoding(input_df, train_encoder=None):
    """Label-encode vehicle_type; fits a new encoder or reuses an existing one."""
    result_df = input_df.copy()

    if train_encoder is None:
        encoder = LabelEncoder()
        result_df['vehicle_type_encoded'] = encoder.fit_transform(result_df['vehicle_type'].astype(str))
        return result_df, encoder
    else:
        result_df['vehicle_type_encoded'] = result_df['vehicle_type'].astype(str).apply(
            lambda x: train_encoder.transform([x])[0] if x in train_encoder.classes_ else -1
        )
        return result_df
