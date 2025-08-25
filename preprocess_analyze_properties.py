
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import holidays
import json
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ML Libraries for DEST functionality
from scipy.stats import zscore, median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.cluster import DBSCAN
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
from numpy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.models import RNNModel, ARIMA as DartsARIMA
from darts.utils.statistics import check_seasonality
from io import StringIO
import requests

# Debug: Check if import worked
print("Available functions:", dir())
try:
    device = get_torch_device()
    print(f"get_torch_device() works: {device}")
except NameError as e:
    print(f"Import failed: {e}")

def classify_range(value, thresholds, labels):
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[i]
    return labels[-1]

# Example usage:
missing_ratio = 0.07
label = classify_range(missing_ratio, thresholds=[0.05, 0.20], labels=['LOW', 'MEDIUM', 'HIGH'])
# returns 'MEDIUM'



# Robust conversion function
def ensure_boolean_anomaly_column(df, column_name='is_anomaly'):
    """Convert anomaly column to boolean regardless of current format"""
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Get unique values to understand the data
    unique_vals = df[column_name].unique()
    
    if df[column_name].dtype == 'bool':
        # Already boolean, nothing to do
        print(f"'{column_name}' is already boolean")
        return df
    
    elif set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({0.0, 1.0}):
        # Numeric 0/1, convert to boolean
        print(f"Converting '{column_name}' from numeric (0/1) to boolean")
        df[column_name] = df[column_name].astype(bool)
        return df
    
    elif set(unique_vals).issubset({'True', 'False', True, False}):
        # String or mixed boolean, convert to boolean
        print(f"Converting '{column_name}' from string/mixed to boolean")
        df[column_name] = df[column_name].astype(bool)
        return df
    
    else:
        # Unexpected values
        print(f"Warning: Unexpected values in '{column_name}': {unique_vals}")
        print("Treating non-zero/non-False values as True")
        df[column_name] = df[column_name].astype(bool)
        return df


def add_advanced_features(df):
    """Add comprehensive features for ML models"""
    df = df.copy()

    df = ensure_boolean_anomaly_column(df, 'is_anomaly')

    # Time-based features (if not already present)
    if 'timestamp' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    
    # Statistical rolling features (user-configurable window)
    for window in [5, 10, 20]:  # User can specify these
        df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        df[f'roll_skew_{window}'] = df['value'].rolling(window).skew()
        df[f'roll_kurt_{window}'] = df['value'].rolling(window).kurt()
        df[f'rolling_slope_{window}'] = df['value'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )

    # Frequency domain features
    if len(df) > 50:
        fft_values = np.fft.fft(df['value'].fillna(df['value'].mean()))
        df['high_pass_5'] = np.abs(fft_values[5:10]).mean() if len(fft_values) > 10 else 0
        df['high_pass_10'] = np.abs(fft_values[10:20]).mean() if len(fft_values) > 20 else 0

    # Lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
        df[f'diff_{lag}'] = df['value'].diff(lag)

    # Statistical measures
    df['iqr_diff'] = df['value'] - df['value'].quantile(0.5)
    df['autocorr_lag1'] = df['value'].autocorr(lag=1) if len(df) > 1 else 0

    return df

def calculate_anomaly_ratio_estimate(df, z_threshold=3.0, multi_column_list=None, combine_method='median'):
    """
    Estimate anomaly ratio based on z-score method for numeric time-series columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time-series data.
    z_threshold : float
        Z-score threshold to mark anomalies.
    multi_column_list : list or None
        If None → use all numeric columns.
        Else → use only the columns specified in the list.
    combine_method : str
        How to combine multi-column ratios: 'median' or 'mean'.

    Returns
    -------
    float
        Combined anomaly ratio across columns.
    dict
        Per-column anomaly ratios.
    """
    # Select columns
    if multi_column_list is None:
        cols = df.select_dtypes(include=[np.number]).columns
    else:
        cols = [c for c in multi_column_list if c in df.columns]

    if 'is_anomaly' in df.columns:
        true_anomaly_ratio = df['is_anomaly'].sum() / len(df)
        anomaly_ratio_estimate = true_anomaly_ratio  # override
        return anomaly_ratio_estimate, 0


    col_ratios = {}
    for col in cols:
        series = df[col].dropna()
        if len(series) == 0:
            col_ratios[col] = np.nan
            continue

        z_scores = np.abs(zscore(series, nan_policy='omit'))
        anomalies = np.sum(z_scores > z_threshold)
        ratio = anomalies / len(series)
        col_ratios[col] = ratio

    valid_ratios = [v for v in col_ratios.values() if not np.isnan(v)]
    if not valid_ratios:
        return np.nan, col_ratios

    if combine_method == 'mean':
        combined = np.mean(valid_ratios)
    else:  # median default
        combined = np.median(valid_ratios)

    return combined, col_ratios


def calculate_stationarity(df):
    try:
        adf_res = adfuller(df['value'].dropna())
        adf_pvalue = adf_res[1]
    except:
        adf_pvalue = 0.5
    return adf_pvalue

def calculate_seasonal_strength_and_dominent_period(df):
    # FFT seasonality detection
    
    try:    
        y = df['value'].fillna(method='ffill') - df['value'].mean()
        amps = np.abs(fft(y))
        peaks = np.argsort(amps)[-3:]
        dominant_period = len(y)/peaks[-1] if len(peaks)>0 else 24
    except: 
        dominant_period = 24
        
    
    # Seasonality strength
    try:
        period = max(2, int(round(dominant_period)))
        if len(df) >= 2 * period:
            res = seasonal_decompose(df['value'], model='additive', period=period, extrapolate_trend='freq')
            seasonal_strength = res.seasonal.std() / (res.trend.std() or 1) 
        else:
            seasonal_strength = 0
    except:
        seasonal_strength = 0
    return seasonal_strength , dominant_period;


def calculate_high_pass(df, window):
    try:
        series = df['value']
        low_pass = series.rolling(window, min_periods=1).mean()
        return (series - low_pass).std()  # <-- std of high-pass
    except:
        return 0

def assess_clustering_potential(props, df):
    """
    Assesses the potential for density-based clustering based on pre-calculated properties.
    """
    # Use the pre-calculated skew and kurtosis from `props`
    skewness_val = props.get('skew')
    kurtosis_val = props.get('kurtosis')
    
    # Define thresholds
    skewness_threshold = 0.5
    kurtosis_threshold = 1.0
    
    # Check for non-uniform distribution
    is_non_uniform = (
        abs(skewness_val) > skewness_threshold or 
        abs(kurtosis_val) > kurtosis_threshold
    )
    
    density_potential = "LOW"
    if is_non_uniform:
        density_potential = "HIGH"
        
    # Optional: Check for multiple modes (peaks) in the histogram of the primary series.
    # This requires the original numerical data, which you can pass to this function.
    primary_series_name = 'value'
    if 'primary_series_name' in df.columns:
        series = df[props.get('primary_series_name')].dropna()
        hist_counts, _ = np.histogram(series, bins='auto')
        peaks = np.sum((hist_counts[1:-1] > hist_counts[:-2]) & (hist_counts[1:-1] > hist_counts[2:]))
        if peaks > 1:
            density_potential = "HIGH"
    
    props['has_density_based_clusters'] = is_non_uniform
    props['density_clustering_potential'] = density_potential
    
    return props



def analyze_data_properties(df, domain_knowledge_available=False, computational_budget='MEDIUM', 
                          real_time_requirement=False, interpretability_needed=True):
    """
    Extended comprehensive analysis of time-series and tabular data properties for rules/models
    
    Added parameters:
    - domain_knowledge_available: Boolean flag for static threshold rules
    - computational_budget: 'LOW'/'MEDIUM'/'HIGH' for model selection
    - real_time_requirement: Boolean for real-time processing needs
    - interpretability_needed: Boolean for explainable models preference
    """
    props = {}
    
    # --- NEW: System/Domain Context ---
    props['domain_knowledge_available'] = domain_knowledge_available
    props['computational_budget'] = computational_budget  # LOW/MEDIUM/HIGH
    props['real_time_requirement'] = real_time_requirement
    props['interpretability_needed'] = interpretability_needed
    
    # --- Timestamp stats ---
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_deltas = df['timestamp'].diff().dropna().dt.total_seconds()
        props['frequency'] = time_deltas.mode().iloc[0] if not time_deltas.empty else None
        props['is_regular'] = time_deltas.nunique() <= 3
        props['missing_timestamps'] = df['timestamp'].isna().sum()
        
        # --- NEW: Temporal Quality Metrics ---
        # Calculate temporal gaps
        if len(time_deltas) > 1:
            expected_interval = time_deltas.mode().iloc[0] if not time_deltas.empty else 3600
            large_gaps = (time_deltas > expected_interval * 2).sum()
            props['temporal_gaps'] = large_gaps
            
            # Data span information
            total_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            props['total_span_hours'] = total_span / 3600
            props['data_density'] = len(df) / (total_span / expected_interval) if total_span > 0 else 1.0
        else:
            props['temporal_gaps'] = 0
            props['total_span_hours'] = 0
            props['data_density'] = 1.0
    else:
        props['frequency'] = None
        props['is_regular'] = None
        props['missing_timestamps'] = 0
        props['temporal_gaps'] = 0
        props['total_span_hours'] = 0
        props['data_density'] = 1.0
    
    # --- Basic stats ---
    props['size'] = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    props['n_features'] = len(numeric_cols)
    props['multi_variate'] = props['n_features'] > 1
    props['missing_value_ratio'] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    
    # --- NEW: Data Quality Metrics ---
    props['completeness_score'] = 1 - props['missing_value_ratio']
    
    # --- NEW: Size Categories (for easier condition checking) ---
    if props['size'] < 500:
        props['size_category'] = 'SMALL'
    elif props['size'] <= 2000:
        props['size_category'] = 'MEDIUM'  
    elif props['size'] <= 10000:
        props['size_category'] = 'MID_LARGE'
    else:
        props['size_category'] = 'LARGE'
    
    # --- If univariate time-series exists ---
    if 'value' in df.columns:
        df['value'] = df['value'].astype(float)
        series = df['value'].dropna()
        
        if len(series) > 0:
            # Central moments
            props['mean'] = series.mean()
            props['std'] = series.std()
            props['skew'] = series.skew()
            props['kurtosis'] = series.kurtosis()
            props['has_negative'] = (series < 0).any()
            props['iqr_diff'] = series.quantile(0.75) - series.quantile(0.25)
            
            # --- NEW: Distribution Analysis ---
            # Normality test (for method selection)
            if len(series) >= 3 and len(series) <= 5000:  # Shapiro-Wilk limitations
                try:
                    _, props['normality_test_pvalue'] = shapiro(series.sample(min(5000, len(series))))
                    props['is_normal'] = props['normality_test_pvalue'] > 0.05
                except:
                    props['normality_test_pvalue'] = 0.0
                    props['is_normal'] = False
            else:
                # Use simpler checks for large datasets
                props['is_normal'] = abs(props['skew']) < 0.5 and props['kurtosis'] < 3.5
                props['normality_test_pvalue'] = 0.5 if props['is_normal'] else 0.01
            
            # Heavy tails indicator
            props['heavy_tails'] = props['kurtosis'] > 4.0
            
            # --- NEW: Volatility and Regime Analysis ---
            if len(series) >= 20:
                # Volatility clustering (GARCH-like behavior)
                returns = series.pct_change().dropna()
                if len(returns) > 10:
                    vol_series = returns.rolling(5, min_periods=3).std()
                    props['volatility_clustering'] = vol_series.autocorr(lag=1) if not vol_series.empty else 0
                else:
                    props['volatility_clustering'] = 0
                
                # Regime changes detection (structural breaks)
                if len(series) >= 50:
                    # Simple regime change detection using rolling statistics
                    window = min(20, len(series) // 4)
                    rolling_mean = series.rolling(window).mean()
                    rolling_std = series.rolling(window).std()
                    
                    mean_changes = abs(rolling_mean.diff()).mean() / props['std'] if props['std'] > 0 else 0
                    std_changes = abs(rolling_std.diff()).mean() / props['std'] if props['std'] > 0 else 0
                    
                    props['regime_changes'] = (mean_changes > 0.1) or (std_changes > 0.1)
                    props['regime_change_intensity'] = mean_changes + std_changes
                else:
                    props['regime_changes'] = False
                    props['regime_change_intensity'] = 0
            else:
                props['volatility_clustering'] = 0
                props['regime_changes'] = False
                props['regime_change_intensity'] = 0
            
            # Trend, seasonality, stationarity  
            props['trend_ratio'] = calculate_trend_strength(df)
            props['adf_pvalue'] = calculate_stationarity(df)
            props['is_stationary'] = props['adf_pvalue'] < 0.05
            props['autocorrelation_score'] = calculate_autocorrelation_score(df)
            
            # Seasonal metrics
            seasonal_strength, dominant_period = calculate_seasonal_strength_and_dominent_period(df)
            props['seasonal_strength'] = seasonal_strength
            props['dominant_period'] = dominant_period
            
            # --- NEW: Enhanced Seasonal Categories ---
            if props['seasonal_strength'] < 0.3:
                props['seasonal_category'] = 'WEAK'
            elif props['seasonal_strength'] < 0.6:
                props['seasonal_category'] = 'MEDIUM'
            else:
                props['seasonal_category'] = 'STRONG'
            
            # Anomaly estimate
            anomaly_ratio, _ = calculate_anomaly_ratio_estimate(df, z_threshold=3.0, multi_column_list=['value'])
            props['anomaly_ratio_estimate'] = anomaly_ratio
            
            # --- NEW: Enhanced Anomaly Categories ---
            if props['anomaly_ratio_estimate'] < 0.05:
                props['anomaly_category'] = 'LOW'
            elif props['anomaly_ratio_estimate'] <= 0.15:
                props['anomaly_category'] = 'MEDIUM'
            else:
                props['anomaly_category'] = 'HIGH'
            
            # Rolling / high-pass for volatility or spiky signal detection
            for window in [5, 10, 20]:
                if len(series) >= window:
                    props[f'roll_skew_{window}'] = series.rolling(window).skew().mean()
                    props[f'roll_kurt_{window}'] = series.rolling(window).kurt().mean()
                    props[f'high_pass_{window}'] = calculate_high_pass(df, window)
                else:
                    props[f'roll_skew_{window}'] = props['skew']
                    props[f'roll_kurt_{window}'] = props['kurtosis']  
                    props[f'high_pass_{window}'] = 0
            
            props['rolling_std_10'] = series.rolling(min(10, len(series))).mean().std() if len(series) >= 10 else props['std']
            
            # Noise level for DL selection
            if len(series) >= 5:
                hp_std = (series - series.rolling(5, min_periods=1).mean()).std()
                props['noise_level'] = hp_std / (props['std'] + 1e-8)
                
                # --- NEW: Noise Categories ---
                if props['noise_level'] < 0.3:
                    props['noise_category'] = 'LOW'
                elif props['noise_level'] <= 0.7:
                    props['noise_category'] = 'MEDIUM'
                else:
                    props['noise_category'] = 'HIGH'
            else:
                props['noise_level'] = 0
                props['noise_category'] = 'LOW'
                
            # --- NEW: Outlier Persistence Analysis ---
            if props['anomaly_ratio_estimate'] > 0.01:  # Only if we have some anomalies
                # Simple persistence check using z-score
                z_scores = np.abs(zscore(series))
                outliers = z_scores > 3
                if outliers.sum() > 0:
                    # Calculate average run length of consecutive outliers
                    runs = []
                    current_run = 0
                    for is_outlier in outliers:
                        if is_outlier:
                            current_run += 1
                        else:
                            if current_run > 0:
                                runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        runs.append(current_run)
                    
                    props['outlier_persistence'] = np.mean(runs) if runs else 1
                    props['outlier_max_persistence'] = max(runs) if runs else 1
                else:
                    props['outlier_persistence'] = 1
                    props['outlier_max_persistence'] = 1
            else:
                props['outlier_persistence'] = 1
                props['outlier_max_persistence'] = 1
        else:
            # Handle empty series
            for key in ['mean', 'std', 'skew', 'kurtosis', 'trend_ratio', 'adf_pvalue', 
                       'autocorrelation_score', 'seasonal_strength', 'dominant_period',
                       'anomaly_ratio_estimate', 'noise_level']:
                props[key] = 0
            props['has_negative'] = False
            props['is_stationary'] = True
            props['is_normal'] = True
    
    # --- Anomaly labels (optional) ---
    if 'is_anomaly' in df.columns:
        props['has_anomaly_labels'] = True
        props['anomaly_ratio'] = df['is_anomaly'].mean()
        
        # --- NEW: Label Quality Analysis ---
        if props['anomaly_ratio'] > 0:
            # Calculate label consistency (how clustered are the anomalies)
            anomaly_indices = df[df['is_anomaly']].index
            if len(anomaly_indices) > 1:
                gaps = np.diff(sorted(anomaly_indices))
                props['anomaly_clustering'] = (gaps == 1).mean()  # Fraction of consecutive anomalies
            else:
                props['anomaly_clustering'] = 0
        else:
            props['anomaly_clustering'] = 0
    else:
        props['has_anomaly_labels'] = False
        props['anomaly_ratio'] = None
        props['anomaly_clustering'] = 0
    
    # --- NEW: Feature Engineering Potential ---
    if 'timestamp' in df.columns and props['is_regular']:
        props['can_use_time_features'] = True
        # Check what time features would be useful
        props['useful_time_features'] = []
        
        if props['total_span_hours'] > 24:  # More than a day
            props['useful_time_features'].append('hour')
        if props['total_span_hours'] > 24 * 7:  # More than a week
            props['useful_time_features'].append('dayofweek')
        if props['total_span_hours'] > 24 * 60:  # More than 2 months
            props['useful_time_features'].append('month')
    else:
        props['can_use_time_features'] = False
        props['useful_time_features'] = []
    
    props = assess_clustering_potential(props, df)
    return props


def analyze_data_properties_v2(df):
    """Comprehensive analysis of time-series and tabular data properties for rules/models"""
    props = {}

    # --- Timestamp stats ---
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_deltas = df['timestamp'].diff().dropna().dt.total_seconds()
        props['frequency'] = time_deltas.mode().iloc[0] if not time_deltas.empty else None  # Used to detect granularity
        props['is_regular'] = time_deltas.nunique() <= 3  # Irregular timestamps = avoid TS models
        props['missing_timestamps'] = df['timestamp'].isna().sum()

    # --- Basic stats ---
    props['size'] = len(df)  # For model complexity and thresholds
    props['n_features'] = len(df.select_dtypes(include=[np.number]).columns)
    props['multi_variate'] = props['n_features'] > 1
    props['missing_value_ratio'] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])

    # --- If univariate time-series exists ---
    if 'value' in df.columns:
        df['value'] = df['value'].astype(float)
        series = df['value'].dropna()

        # Central moments
        props['mean'] = series.mean()
        props['std'] = series.std()
        props['skew'] = series.skew()  # For detecting asymmetry (useful for threshold or log-transform)
        props['kurtosis'] = series.kurtosis()  # Spiky or heavy-tailed data
        props['has_negative'] = (series < 0).any()
        props['iqr_diff'] = series.quantile(0.75) - series.quantile(0.25)  # For IQR-based rule

        # Trend, seasonality, stationarity
        props['trend_ratio'] = calculate_trend_strength(df)  # Model: ARIMA, Rule: trend_rule
        props['adf_pvalue'] = calculate_stationarity(df)     # Stationarity score
        props['is_stationary'] = props['adf_pvalue'] < 0.05
        props['autocorrelation_score'] = calculate_autocorrelation_score(df)  # For TS model auto-selection

        # Seasonal metrics
        seasonal_strength, dominant_period = calculate_seasonal_strength_and_dominent_period(df)
        props['seasonal_strength'] = seasonal_strength  # Model: Prophet, SARIMA
        props['dominant_period'] = dominant_period      # Interpretability, visualizations

        # Anomaly estimate
        anomaly_ratio, _ = calculate_anomaly_ratio_estimate(df, z_threshold=3.0, multi_column_list=['value'])
        props['anomaly_ratio_estimate'] = anomaly_ratio

        # Rolling / high-pass for volatility or spiky signal detection
        for window in [5, 10, 20]:
            props[f'roll_skew_{window}'] = series.rolling(window).skew().mean()
            props[f'roll_kurt_{window}'] = series.rolling(window).kurt().mean()
            props[f'high_pass_{window}'] = calculate_high_pass(df, window)
        props['rolling_std_10'] = series.rolling(10).mean().std()

        # Noise level for DL selection
        hp_std = (series - series.rolling(5, min_periods=1).mean()).std()
        props['noise_level'] = hp_std / (props['std'] + 1e-8)

    # --- Anomaly labels (optional) ---
    if 'is_anomaly' in df.columns:
        props['has_anomaly_labels'] = True
        props['anomaly_ratio'] = df['is_anomaly'].mean()
    else:
        props['has_anomaly_labels'] = False
        props['anomaly_ratio'] = None

    return props


def analyze_data_properties_old( df):
    """Comprehensive data properties analysis"""
    seasonal_strength, dominent_period =  (calculate_seasonal_strength_and_dominent_period(df) if 'value' in df.columns else (0,0))
    anomaly_ratio,_ = calculate_anomaly_ratio_estimate(df, z_threshold=3.0, multi_column_list=['value'])
    props = {
        # Basic properties
        'size': len(df),
        'n_features': len(df.select_dtypes(include=[np.number]).columns),
        'feature_dimensions': df.select_dtypes(include=[np.number]).shape[1],
        'multi_variate': df.select_dtypes(include=[np.number]).shape[1] > 1,
        'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
    
        # Statistical properties
        'mean': df['value'].mean() if 'value' in df.columns else 0,
        'std': df['value'].std() if 'value' in df.columns else 0,
        'trend_ratio': calculate_trend_strength(df) if 'value' in df.columns else 0,
        'iqr_diff': df['value'].quantile(0.75) - df['value'].quantile(0.25) if 'value' in df.columns else 0,
    
        # Time series properties
        'autocorr_lag1': df['value'].autocorr(lag=1) if 'value' in df.columns and len(df) > 1 else 0,
        'autocorrelation_score': calculate_autocorrelation_score(df) if 'value' in df.columns else 0,
        'adf_pvalue': calculate_stationarity(df) if 'value' in df.columns else 1.0,
        'seasonal_strength': seasonal_strength,
        'dominant_period': dominent_period,
    
        # Anomaly characteristics
        'anomaly_ratio_estimate': anomaly_ratio if 'value' in df.columns else 0.05,
    
        # Rolling statistics
        'rolling_mean': df['value'].rolling(10).mean().std() if 'value' in df.columns else 0,
    }

    # Add user-configurable rolling features
    for window in [5, 10, 20]:
        props[f'roll_skew_{window}'] = df['value'].rolling(window).skew().mean() if 'value' in df.columns else 0
        props[f'roll_kurt_{window}'] = df['value'].rolling(window).kurt().mean() if 'value' in df.columns else 0
        props[f'high_pass_{window}'] = calculate_high_pass(df, window) if 'value' in df.columns else 0

    return props

def calculate_trend_strength( df):
    """Calculate trend strength"""
    if len(df) < 10:
        return 0
    x = np.arange(len(df))
    slope, _, r_value, _, _ = stats.linregress(x, df['value'])
    return abs(r_value)

def calculate_autocorrelation_score(df):
    """Calculate overall autocorrelation strength"""
    autocorr_values = [df['value'].autocorr(lag=i) for i in range(1, min(20, len(df)//4))]
    return np.mean([abs(x) for x in autocorr_values if not np.isnan(x)])

# === MODEL SELECTION ENGINE ===

def calculate_score(method_conditions, data_properties, base_priority):
    """
    Calculates a weighted score for a given method based on data properties.

    Args:
        method_conditions (list): List of condition strings for the method.
        data_properties (dict): Dictionary of calculated data properties.
        base_priority (int): The base numeric priority of the method.

    Returns:
        float: The total weighted score.
    """
    total_score = float(base_priority)
    for condition_str in method_conditions:
        try:
            # Check for specific condition string matches in weights
            if condition_str in CONDITION_WEIGHTS:
                if eval(condition_str, {}, data_properties):
                    total_score += CONDITION_WEIGHTS[condition_str]
            else:
                # Handle conditions not explicitly in CONDITION_WEIGHTS,
                # like complex boolean expressions.
                if eval(condition_str, {}, data_properties):
                    total_score += 10 # A default small boost for a match
        except Exception as e:
            # Handle cases where a property might be missing
            # or the condition is invalid for this data.
            continue
            
    return total_score


def evaluate_model_conditions(props, model_config):
    """
    Evaluate if a model's conditions are met given data properties
    """
    conditions = model_config.get('conditions', [])
    if isinstance(conditions, str):
        conditions = [conditions]
    
    for condition in conditions:
        try:
            # Replace property names with actual values
            eval_condition = condition
            for prop_name, prop_value in props.items():
                if prop_name in eval_condition:
                    if isinstance(prop_value, bool):
                        eval_condition = eval_condition.replace(f'{prop_name}', str(prop_value))
                    elif isinstance(prop_value, str):
                        eval_condition = eval_condition.replace(f'{prop_name}', f"'{prop_value}'")
                    else:
                        eval_condition = eval_condition.replace(f'{prop_name}', str(prop_value))
            
            # Evaluate the condition
            if not eval(eval_condition):
                return False
        except Exception as e:
            print(f"Error evaluating condition '{condition}': {e}")
            return False
    
    return True

def select_recommended_methods(props, include_rules=True, include_ml=True, max_methods=5):
    """
    Select recommended methods based on data properties
    
    Returns:
    - recommended_methods: List of method names with scores and metadata
    - data_summary: Summary of key data characteristics for UI
    """
    
    from enhanced_model_mappings import RULES_MAPPING, ML_MAPPING
    
    recommendations = []
    
    # Evaluate rules
    if include_rules:
        for method_name, config in RULES_MAPPING.items():
            print(f" RULE : method_name is : {method_name} ") 
            
            # if evaluate_model_conditions(props, config):
            #    score = calculate_method_score(props, config)
            method_config = RULES_MAPPING.get(method_name)
            score, requried_conditions_met = calculate_final_score_and_evaluate(props, method_config)
            if requried_conditions_met:
                recommendations.append({
                    'method': method_name,
                    'type': 'RULE',
                    'priority': config.get('priority', 'MEDIUM'),
                    'score': score,
                    'computational_cost': config.get('computational_cost', 'LOW'),
                    'interpretable': True,
                    'real_time_capable': True
                })
            else:
                print(f" RULE : method_name is : {method_name} is not selected after model conditions evaluation")
    
    # Evaluate ML methods
    if include_ml:
        for method_name, config in ML_MAPPING.items():
            print(f" ML : method_name is : {method_name} ") 
            #if evaluate_model_conditions(props, config):
            #    score = calculate_method_score(props, config)
            method_config = ML_MAPPING.get(method_name)
            score, requried_conditions_met = calculate_final_score_and_evaluate(props, method_config)

            if requried_conditions_met:
                recommendations.append({
                    'method': method_name,
                    'type': 'ML',
                    'priority': config.get('priority', 'MEDIUM'),
                    'score': score,
                    'computational_cost': config.get('computational_cost', 'MEDIUM'),
                    'interpretable': method_name in ['zscore', 'mad', 'ewma', 'prophet'],
                    'real_time_capable': method_name not in ['lstm', 'gru', 'autoencoder', 'usad']
                })

            else:
                print(f" ML : method_name is : {method_name} is not selected after model conditions evaluation")
    
    print(f"Methods for recommendations are : {recommendations}")
    # Filter based on system constraints
    filtered_recommendations = filter_by_constraints(recommendations, props)
    
    # Sort by score and priority
    priority_weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    for rec in filtered_recommendations:
        rec['final_score'] = rec['score'] * priority_weights.get(rec['priority'], 2)
    
    # Sort and limit
    filtered_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    top_recommendations = filtered_recommendations[:max_methods]
    
    # Create data summary for UI
    data_summary = create_data_summary_for_ui(props)
    
    return top_recommendations, data_summary

def calculate_method_score(props, config):
    """Calculate a score for how well a method fits the data"""
    base_score = 50  # Base score
    
    # Boost score based on priority
    priority = config.get('priority', 'MEDIUM')
    if priority == 'HIGH':
        base_score += 20
    elif priority == 'LOW':
        base_score -= 10
    
    # Data size bonus/penalty
    size = props.get('size', 0)
    if size < 100:
        base_score -= 15  # Too small for most methods
    elif size > 10000:
        base_score += 10  # Large dataset bonus
    
    # Data quality bonus/penalty
    completeness = props.get('completeness_score', 1.0)
    base_score += (completeness - 0.8) * 25  # Penalty for missing data
    
    # Specific bonuses
    if props.get('seasonal_strength', 0) > 0.8:
        if 'seasonal' in config.get('method', ''):
            base_score += 15
    
    if props.get('anomaly_ratio_estimate', 0) > 0.1:
        if config.get('method', '') in ['isolation_forest', 'dbscan']:
            base_score += 10

    if props.get('density_clustering_potential') == 'HIGH':
        if config.get('method', '') in ['dbscan', 'local_outlier_factor']:
            base_score += 25  # Add a significant bonus

    
    return max(0, min(100, base_score))  # Clamp between 0-100

def calculate_final_score_and_evaluate(props, method_config):
    """
    Evaluates all conditions for a method using their severity and weight, 
    and calculates a final score.

    Args:
        props (dict): Dictionary of data properties.
        method_config (dict): Configuration for the method, including conditions.

    Returns:
        tuple: (final_score, all_critical_conditions_met)
    """
    conditions_list = method_config.get('conditions', [])
    base_priority = method_config.get('base_priority', 50)
    
    total_score = float(base_priority)
    all_critical_conditions_met = True

    for condition_item in conditions_list:
        condition_str = condition_item['condition']
        severity = condition_item['severity']
        weight = condition_item['weight']

        try:
            condition_met = eval(condition_str, {}, props)

            if condition_met:
                total_score += weight
            else:
                if severity == 'must':
                    all_critical_conditions_met = False
                    total_score -= weight * 2 # Heavy penalty for failing a 'must' condition
                elif severity == 'optional':
                    total_score -= weight * 0.5 # Lighter penalty for failing an 'optional' condition

        except Exception as e:
            # Handle cases where a property might be missing or condition is invalid
            print(f"Error evaluating condition '{condition_str}': {e}")
            all_critical_conditions_met = False
            total_score -= 50 # Severe penalty for an error
    
    return max(0, total_score), all_critical_conditions_met


def apply_meta_priorities(recommendations, props):
    filtered_recs = []
    
    # 1. Apply Computational and Interpretability Constraints (Filtering)
    real_time_needed = props.get('real_time_requirement', False)
    interpretability_needed = props.get('interpretability_needed', True)
    
    for rec in recommendations:
        # Check real-time constraint
        if real_time_needed and not rec.get('real_time_capable', True):
            continue

        # Check interpretability constraint (penalty, not filter)
        if interpretability_needed and not rec.get('interpretable', False):
            rec['score'] *= 0.8  # Apply a penalty to the score

        filtered_recs.append(rec)
    
    # 2. Apply Data-Driven Priorities (Score Boosting)
    final_recs = []
    for rec in filtered_recs:
        current_score = rec['score']
        
        # Apply data-driven boosts
        if props.get('seasonal_strength', 0) > 0.8 and 'seasonal' in rec['method']:
            current_score += 25 # Strongly prefer seasonal methods

        if props.get('trend_ratio', 0) > 0.6 and 'trend' in rec['method']:
            current_score += 15 # Prefer trend-aware methods

        # Add more rules as needed
        
        rec['score'] = current_score
        final_recs.append(rec)

    # Sort and return
    final_recs.sort(key=lambda x: x['score'], reverse=True)
    return final_recs


def filter_by_constraints(recommendations, props):
    """Filter recommendations based on system constraints"""
    filtered = []
    
    computational_budget = props.get('computational_budget', 'MEDIUM')
    real_time_requirement = props.get('real_time_requirement', False)
    interpretability_needed = props.get('interpretability_needed', True)
    
    for rec in recommendations:
        # Check computational constraints
        cost = rec.get('computational_cost', 'MEDIUM')
        if computational_budget == 'LOW' and cost in ['HIGH', 'VERY_HIGH']:
            continue
        if computational_budget == 'MEDIUM' and cost == 'VERY_HIGH':
            continue
            
        # Check real-time constraints
        if real_time_requirement and not rec.get('real_time_capable', True):
            continue
            
        # Check interpretability constraints  
        if interpretability_needed and not rec.get('interpretable', False):
            rec['score'] *= 0.8  # Penalty but don't exclude
            
        filtered.append(rec)
    
    return filtered

def create_data_summary_for_ui(props):
    """Create a summary of data properties for UI display"""
    summary = {
        'basic_info': {
            'size': props.get('size', 0),
            'size_category': props.get('size_category', 'UNKNOWN'),
            'n_features': props.get('n_features', 0),
            'multivariate': props.get('multi_variate', False),
            'completeness': f"{props.get('completeness_score', 0) * 100:.1f}%"
        },
        'time_series_info': {
            'is_regular': props.get('is_regular', False),
            'frequency_seconds': props.get('frequency'),
            'total_span_hours': props.get('total_span_hours', 0),
            'temporal_gaps': props.get('temporal_gaps', 0)
        },
        'statistical_properties': {
            'distribution': 'Normal' if props.get('is_normal', False) else 'Non-normal',
            'skewness': props.get('skew', 0),
            'kurtosis': props.get('kurtosis', 0),
            'has_negative_values': props.get('has_negative', False),
            'noise_level': props.get('noise_category', 'UNKNOWN')
        },
        'pattern_analysis': {
            'trend_strength': props.get('trend_ratio', 0),
            'seasonal_strength': props.get('seasonal_category', 'UNKNOWN'), 
            'stationarity': 'Stationary' if props.get('is_stationary', False) else 'Non-stationary',
            'autocorrelation': props.get('autocorrelation_score', 0),
            'dominant_period': props.get('dominant_period', 0)
        },
        'anomaly_characteristics': {
            'estimated_anomaly_ratio': f"{props.get('anomaly_ratio_estimate', 0) * 100:.2f}%",
            'anomaly_category': props.get('anomaly_category', 'UNKNOWN'),
            'has_labels': props.get('has_anomaly_labels', False),
            'outlier_persistence': props.get('outlier_persistence', 1)
        },
        'recommendations_context': {
            'domain_knowledge_available': props.get('domain_knowledge_available', False),
            'computational_budget': props.get('computational_budget', 'MEDIUM'),
            'real_time_requirement': props.get('real_time_requirement', False),
            'interpretability_needed': props.get('interpretability_needed', True)
        }
    }
    
    return summary

PROPERTY_TOOLTIPS = {
    # === TIMESTAMP PROPERTIES ===
    'frequency': "Time interval between consecutive data points in seconds (e.g., 3600 = hourly data)",
    'is_regular': "Whether data points are collected at consistent time intervals (True = regular, False = irregular)",
    'missing_timestamps': "Number of missing or invalid timestamp values in the dataset",
    'temporal_gaps': "Number of large gaps in the time series (missing data periods longer than expected)",
    'total_span_hours': "Total time span covered by the dataset in hours",
    'data_density': "Ratio of actual data points to expected data points (1.0 = no gaps, <1.0 = sparse data)",
    
    # === BASIC STATISTICS ===
    'size': "Total number of data points in the dataset",
    'size_category': "Dataset size classification: SMALL (<500), MEDIUM (≤2000), MID_LARGE (≤10000), LARGE (>10000)",
    'n_features': "Number of numeric columns/features in the dataset",
    'multi_variate': "Whether the dataset has multiple numeric features (True = multivariate, False = univariate)",
    'missing_value_ratio': "Percentage of missing values across all columns (0.0 = no missing data, 1.0 = all missing)",
    'completeness_score': "Data quality score based on non-missing values (1.0 = complete, 0.0 = all missing)",
    
    # === DISTRIBUTION PROPERTIES ===
    'mean': "Average value of the time series (central tendency measure)",
    'std': "Standard deviation showing data spread around the mean (higher = more variable)",
    'skew': "Data asymmetry measure (0 = symmetric, positive = right-skewed, negative = left-skewed)",
    'kurtosis': "Tail heaviness measure (3 = normal distribution, >3 = heavy tails, <3 = light tails)",
    'has_negative': "Whether the dataset contains negative values (important for certain models)",
    'iqr_diff': "Interquartile range (75th - 25th percentile), robust measure of data spread",
    'normality_test_pvalue': "Statistical test result for normal distribution (>0.05 suggests normal distribution)",
    'is_normal': "Whether data follows approximately normal distribution (affects model selection)",
    'heavy_tails': "Whether data has unusually heavy tails indicating extreme values",
    
    # === TREND AND STATIONARITY ===
    'trend_ratio': "Strength of linear trend in data (0 = no trend, 1 = perfect linear trend)",
    'adf_pvalue': "Augmented Dickey-Fuller test p-value for stationarity (<0.05 suggests stationary)",
    'is_stationary': "Whether the time series has constant mean and variance over time",
    'autocorrelation_score': "Average correlation between data points and their past values (higher = more predictable)",
    'regime_changes': "Whether the data shows structural breaks or changing patterns over time",
    'regime_change_intensity': "Strength of structural changes in the data (higher = more regime shifts)",
    
    # === SEASONALITY PROPERTIES ===
    'seasonal_strength': "Strength of repeating patterns in data (0 = no seasonality, 1 = perfect seasonality)",
    'seasonal_category': "Seasonality classification: WEAK (<0.3), MEDIUM (0.3-0.6), STRONG (>0.6)",
    'dominant_period': "Length of the main repeating cycle (e.g., 24 for daily patterns, 7 for weekly)",
    'useful_time_features': "List of datetime features that would help models (hour, dayofweek, month)",
    'can_use_time_features': "Whether the dataset supports time-based feature engineering",
    
    # === ANOMALY CHARACTERISTICS ===
    'anomaly_ratio_estimate': "Estimated percentage of anomalous data points using statistical methods",
    'anomaly_category': "Anomaly density classification: LOW (<5%), MEDIUM (5-15%), HIGH (>15%)",
    'anomaly_ratio': "Actual percentage of labeled anomalies (when ground truth labels exist)",
    'has_anomaly_labels': "Whether the dataset includes ground truth anomaly labels",
    'anomaly_clustering': "Tendency of anomalies to occur consecutively (1.0 = always consecutive, 0.0 = isolated)",
    'outlier_persistence': "Average duration of anomalous periods (1 = single points, >1 = persistent anomalies)",
    'outlier_max_persistence': "Maximum consecutive length of anomalous periods in the dataset",
    
    # === VOLATILITY AND NOISE ===
    'noise_level': "Amount of random variation relative to signal strength (higher = noisier data)",
    'noise_category': "Noise classification: LOW (<0.3), MEDIUM (0.3-0.7), HIGH (>0.7)",
    'volatility_clustering': "Tendency for high/low volatility periods to cluster together (GARCH-like behavior)",
    'rolling_std_10': "Standard deviation of 10-point rolling averages (measures local variability)",
    
    # === ROLLING WINDOW STATISTICS ===
    'roll_skew_5': "Average skewness calculated over 5-point rolling windows (local asymmetry)",
    'roll_skew_10': "Average skewness calculated over 10-point rolling windows (medium-term asymmetry)",
    'roll_skew_20': "Average skewness calculated over 20-point rolling windows (long-term asymmetry)",
    'roll_kurt_5': "Average kurtosis calculated over 5-point rolling windows (local tail behavior)",
    'roll_kurt_10': "Average kurtosis calculated over 10-point rolling windows (medium-term spikiness)",
    'roll_kurt_20': "Average kurtosis calculated over 20-point rolling windows (long-term tail behavior)",
    'high_pass_5': "High-frequency variation after removing 5-point trend (short-term noise)",
    'high_pass_10': "High-frequency variation after removing 10-point trend (medium-term noise)",
    'high_pass_20': "High-frequency variation after removing 20-point trend (long-term deviations)",
    
    # === SYSTEM CONSTRAINTS ===
    'domain_knowledge_available': "Whether expert knowledge about normal data ranges is available",
    'computational_budget': "Available computing resources: LOW (basic), MEDIUM (standard), HIGH (advanced)",
    'real_time_requirement': "Whether anomaly detection must run in real-time with low latency",
    'interpretability_needed': "Whether the anomaly detection method must provide explainable results",
    
    # === MODEL RECOMMENDATION METADATA ===
    'method': "Name of the recommended anomaly detection method or rule",
    'type': "Method category: RULE (threshold-based) or ML (machine learning)",
    'priority': "Method preference level: HIGH (strongly recommended), MEDIUM (suitable), LOW (fallback)",
    'score': "Numerical score indicating how well the method fits the data characteristics",
    'final_score': "Priority-weighted score used for final ranking and selection",
    'computational_cost': "Resource requirements: LOW (fast), MEDIUM (moderate), HIGH (intensive), VERY_HIGH (heavy)",
    'interpretable': "Whether the method provides easily understandable explanations for its decisions",
    'real_time_capable': "Whether the method can process data streams with minimal latency",
    
    # === UI DATA SUMMARY SECTIONS ===
    'basic_info': "Fundamental dataset characteristics: size, features, completeness",
    'time_series_info': "Temporal data properties: regularity, frequency, gaps",
    'statistical_properties': "Data distribution characteristics: normality, skewness, noise",
    'pattern_analysis': "Temporal patterns: trends, seasonality, stationarity, correlations",
    'anomaly_characteristics': "Anomaly-related properties: estimated ratios, persistence, labels",
    'recommendations_context': "System constraints and preferences affecting method selection"
}

# === CATEGORY DESCRIPTIONS ===
CATEGORY_TOOLTIPS = {
    'SMALL': "Dataset with fewer than 500 data points - suitable for simple methods",
    'MEDIUM': "Dataset with 500-2000 points - can use moderate complexity methods", 
    'MID_LARGE': "Dataset with 2000-10000 points - supports advanced statistical methods",
    'LARGE': "Dataset with 10000+ points - enables deep learning and complex models",
    
    'WEAK': "Low strength pattern - may not be reliable for model-based methods",
    'MEDIUM': "Moderate strength pattern - suitable for most methods with caution",
    'STRONG': "High strength pattern - excellent for specialized methods",
    
    'LOW': "Minimal level - ideal conditions for sensitive methods",
    'MEDIUM': "Moderate level - standard conditions requiring robust methods",
    'HIGH': "Elevated level - challenging conditions requiring specialized approaches",
    
    'RULE': "Threshold or pattern-based method using predefined logic",
    'ML': "Machine learning method that learns patterns from data",
    
    'Normal': "Data follows bell-curve distribution - suitable for parametric methods",
    'Non-normal': "Data deviates from bell-curve - requires non-parametric or robust methods",
    
    'Stationary': "Data properties remain constant over time - suitable for classical time series methods",
    'Non-stationary': "Data properties change over time - may need preprocessing or adaptive methods"
}

# === USAGE HELPER FUNCTIONS ===

def get_tooltip(property_name):
    """Get tooltip text for a specific property"""
    return PROPERTY_TOOLTIPS.get(property_name, f"Description not available for '{property_name}'")

def get_category_tooltip(category_value):
    """Get tooltip text for a category value"""
    return CATEGORY_TOOLTIPS.get(category_value, f"Description not available for '{category_value}'")

def get_all_tooltips_for_ui():
    """Return all tooltips formatted for UI consumption"""
    return {
        'properties': PROPERTY_TOOLTIPS,
        'categories': CATEGORY_TOOLTIPS
    }

def create_tooltip_mapping_for_data_summary(data_summary):
    """Create tooltip mapping specifically for the data summary structure"""
    tooltip_mapping = {}
    
    for section_name, section_data in data_summary.items():
        tooltip_mapping[section_name] = {
            'section_tooltip': get_tooltip(section_name),
            'property_tooltips': {}
        }
        
        for prop_name, prop_value in section_data.items():
            tooltip_mapping[section_name]['property_tooltips'][prop_name] = {
                'property_tooltip': get_tooltip(prop_name),
                'value_tooltip': get_category_tooltip(prop_value) if isinstance(prop_value, str) else None
            }
    
    return tooltip_mapping

# === EXAMPLE USAGE ===

def demonstrate_tooltip_usage():
    """Example of how to use tooltips in your UI"""
    
    # Example data summary (from your analysis)
    data_summary = {
        'basic_info': {
            'size': 2000,
            'size_category': 'MID_LARGE',
            'n_features': 1,
            'multivariate': False,
            'completeness': '98.5%'
        },
        'statistical_properties': {
            'distribution': 'Non-normal',
            'skewness': 1.2,
            'noise_level': 'MEDIUM'
        }
    }
    
    # Get tooltips for UI
    tooltips = create_tooltip_mapping_for_data_summary(data_summary)
    
    # Example: How to display in UI
    print("=== TOOLTIP USAGE EXAMPLE ===")
    for section, content in tooltips.items():
        print(f"\nSection: {section}")
        print(f"Section Help: {content['section_tooltip']}")
        
        for prop, tooltip_data in content['property_tooltips'].items():
            print(f"  Property: {prop}")
            print(f"  Help: {tooltip_data['property_tooltip']}")
            if tooltip_data['value_tooltip']:
                print(f"  Value Help: {tooltip_data['value_tooltip']}")

