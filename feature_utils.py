# feature_utils.py
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np

def get_scaled_features_enhanced(df, scaling_method='standard'):
    """Enhanced feature scaling with multiple methods"""
    feature_columns = ['value', 'hour', 'day_of_week', 'hour_sin', 'hour_cos',
                      'day_sin', 'day_cos', 'week_of_year', 'month', 'day_of_month']
        
    bool_features = ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']
    available_features = [col for col in feature_columns + bool_features if col in df.columns]
        
    X = df[available_features].copy()
    for col in bool_features:
        if col in X.columns:
            X[col] = X[col].astype(int)
        
    X = X.fillna(X.mean())
        
    if scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:   
        scaler = StandardScaler()
        
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, available_features

def _prepare_training_data(df, train_ratio=0.8, use_anomaly_labels=True):
    """Prepare training data while preserving temporal order"""
    if use_anomaly_labels and 'is_anomaly' in df.columns:
        # Find the largest consecutive normal sequence from the beginning
        normal_mask = ~df['is_anomaly']
        consecutive_normal = 0
        max_consecutive = 0
            
        for i, is_normal in enumerate(normal_mask):
            if is_normal:
                consecutive_normal += 1
                max_consecutive = max(max_consecutive, consecutive_normal)
            else:
                consecutive_normal = 0
                
        # Use up to train_ratio of data, but ensure it's mostly normal
        train_size = min(int(train_ratio * len(df)), max_consecutive)
        train_df = df.iloc[:train_size]
    else:
        train_size = int(train_ratio * len(df))
        train_df = df.iloc[:train_size]
        
    return train_df, train_size

@staticmethod
def _adaptive_threshold(errors, method='dynamic_percentile', base_percentile=95.0):
    """Improved threshold calculation with multiple methods"""
    # Now errors should be a clean list/array of scalar values
    errors_array = np.array(errors)
    errors_clean = errors_array[~np.isnan(errors_array) & (errors_array > 0)]
        
    if method == 'dynamic_percentile':
        # Adjust percentile based on error distribution
        q75, q95 = np.percentile(errors_clean, [75, 95])
        if q95 / q75 > 3:  # High variance - use lower percentile
            percentile = base_percentile - 5
        else:
            percentile = base_percentile
        return np.percentile(errors_clean, percentile)
            
    elif method == 'iqr_outlier':
        Q1 = np.percentile(errors_clean, 25)
        Q3 = np.percentile(errors_clean, 75)
        IQR = Q3 - Q1
        return Q3 + 1.5 * IQR
            
    elif method == 'mad':  # Median Absolute Deviation
        median = np.median(errors_clean)
        mad = np.median(np.abs(errors_clean - median))
        return median + 3 * mad
            
    else:  # fallback to percentile
        return np.percentile(errors_clean, base_percentile)


def _adaptive_threshold(errors, method='dynamic_percentile', base_percentile=95.0):
    """Improved threshold calculation with multiple methods"""

    # First, convert the array to a float data type
    errors_float = np.array(errors, dtype=np.float64)

    # Assuming 'errors' is a NumPy array or a pandas Series
    # First, remove all NaN values from the array
    errors_no_nan = errors[~np.isnan(errors)]

    # Now, filter out the padding zeros from the cleaned array
    errors_clean = errors_no_nan[errors_no_nan > 0] 


    if method == 'dynamic_percentile':
        # Adjust percentile based on error distribution
        q75, q95 = np.percentile(errors_clean, [75, 95])
        if q95 / q75 > 3:  # High variance - use lower percentile
            percentile = base_percentile - 5
        else:
            percentile = base_percentile
        return np.percentile(errors_clean, percentile)
            
    elif method == 'iqr_outlier':
        Q1 = np.percentile(errors_clean, 25)
        Q3 = np.percentile(errors_clean, 75)
        IQR = Q3 - Q1
        return Q3 + 1.5 * IQR
            
    elif method == 'mad':  # Median Absolute Deviation
        median = np.median(errors_clean)
        mad = np.median(np.abs(errors_clean - median))
        return median + 3 * mad
            
    else:  # fallback to percentile
        return np.percentile(errors_clean, base_percentile)

@staticmethod
def _detect_datetime_features(df, timestamp_col='timestamp'):
    """Detect available datetime features from timestamp column"""
    encoders = {
        'cyclic': {'future': []},
        'datetime_attribute': {'future': []}
    }
        
    if timestamp_col not in df.columns:
        print(f"Warning: {timestamp_col} column not found, skipping datetime encoders")
        return None
            
    try:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            timestamp_series = pd.to_datetime(df[timestamp_col])
        else:
            timestamp_series = df[timestamp_col]
            
        # Check what datetime components are available and useful
        sample_timestamps = timestamp_series.dropna()
        if len(sample_timestamps) == 0:
            return None
                
        # Check if we have sub-daily resolution (useful for hour encoding)
        time_diff = sample_timestamps.iloc[-1] - sample_timestamps.iloc[0]
        has_hourly_resolution = len(sample_timestamps) > 1 and (time_diff.total_seconds() / len(sample_timestamps) < 24 * 3600)
            
        # Check if we have multi-day data (useful for day-of-week encoding)
        has_multi_day = time_diff.days > 1
            
        # Add useful encoders based on data characteristics
        if has_hourly_resolution:
            encoders['cyclic']['future'].append('hour')
            encoders['datetime_attribute']['future'].append('hour')
            print("Added hour encoding (detected sub-daily resolution)")
                
        if has_multi_day:
            encoders['cyclic']['future'].append('dayofweek')
            encoders['datetime_attribute']['future'].append('dayofweek')
            print("Added day-of-week encoding (detected multi-day data)")
                
        # Add month encoding for longer time series (useful for seasonal patterns)
        if time_diff.days > 60:  # More than 2 months
            encoders['cyclic']['future'].append('month')
            encoders['datetime_attribute']['future'].append('month')
            print("Added month encoding (detected seasonal patterns)")
            
        # Remove empty encoders
        if not encoders['cyclic']['future']:
            encoders.pop('cyclic')
        if not encoders['datetime_attribute']['future']:
            encoders.pop('datetime_attribute')
                
        return encoders if encoders else None
        
    except Exception as e:
        print(f"Error detecting datetime features: {e}")
        return None
