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
from numpy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.models import RNNModel, ARIMA as DartsARIMA
from darts.utils.statistics import check_seasonality
from io import StringIO
import requests
from additional_anomaly_methods import apply_kmeans_detection, train_usad, usad_score, meta_aad_active_loop, get_torch_device


class RuleBasedAnomaly:
    def __init__(self):
        all_rules = {}

    @staticmethod
    def get_all_rule_methods(props):
        """Get complete list of rule-based methods with intelligent recommendations"""
        all_rules = {
            # Statistical rules
            #'zscore_rule': {'enabled': True, 'category': 'statistical'},
            #'iqr_rule': {'enabled': True, 'category': 'statistical'},
            #'mad_rule': {'enabled': False, 'category': 'statistical'},
            #'moving_avg_rule': {'enabled': True, 'category': 'statistical'},

            # threshold-based rules
            'static_min_threshold': {'enabled': False, 'category': 'threshold'},
            'static_max_threshold': {'enabled': False, 'category': 'threshold'},
            'static_range_rule': {'enabled': False, 'category': 'threshold'},
            
            # percentage-based rules
            'static_percentage_rule': {'enabled': False, 'category': 'percentage'},
            'percentile_rule': {'enabled': False, 'category': 'percentage'},
        
            # Pattern-based rules
            'sharp_jump_rule': {'enabled': False, 'category': 'pattern'},
            'rate_of_change': {'enabled': False, 'category': 'pattern'},
            'trend_rule': {'enabled': False, 'category': 'pattern'},
            'baseline_deviation_rule': {'enabled': False, 'category': 'pattern'},
        
            # Seasonal rules
            'seasonal_decomp_rule': {'enabled': False, 'category': 'seasonal'},
            'periodic_rule': {'enabled': False, 'category': 'seasonal'},
 
            # contextual rules
            'consecutive_anomaly':{'enabled': False, 'category': 'chained'},
        }
    
        '''
        # Auto-enable based on properties
        if props.get('trend_ratio', 0) > 0.7:
            all_rules['trend_rule']['enabled'] = True
        
        if props.get('seasonal_strength', 0) > 0.5:
            all_rules['seasonal_decomp_rule']['enabled'] = True
        
        if props.get('autocorrelation_score', 0) > 0.6:
            all_rules['sharp_jump_rule']['enabled'] = True
        ''' 
        return all_rules

    @staticmethod
    def get_rule_category_emoji(rule_category):
        if rule_category == "threshold":
            return " ⚖️  "
        elif rule_category == "percentage":
            return " 🎯 "
        elif rule_category == "pattern":
            return " 🧩 "
        elif rule_category == "seasonal":
            return " 🗓️ "
        elif rule_category == "chained":
            return " 🔗 "
        else:
            return ""

    # Default configurations for rule-based methods
    RULE_DEFAULTS = {
        'sharp_jump_rule': {
            'jump_threshold': 2.0,
            'min_jump_threshold': 1.0,
            'max_jump_threshold': 5.0,
            'step': 0.1
        },
        'trend_rule': {
            'trend_window': 10,
            'min_trend_window': 5,
            'max_trend_window': 50,
            'step': 1,
            'trend_threshold': 0.8
        },
        'baseline_deviation_rule': {
            'baseline_window': 50,
            'min_baseline_window': 10,
            'max_baseline_window': 200,
            'step': 5,
            'deviation_threshold': 2.0
        },
        'zscore_threshold': {
            'zscore_value': 3.0,
            'min_zscore': 1.0,
            'max_zscore': 5.0,
            'step': 0.1,
            'use_modified': False
        },
        'iqr_method': {
            'iqr_multiplier': 1.5,
            'min_iqr_multiplier': 1.0,
            'max_iqr_multiplier': 3.0,
            'step': 0.1,
            'use_median': False
        },
        'moving_avg': {
            'moving_window': 10,
            'min_moving_window': 3,
            'max_moving_window': 50,
            'step': 1,
            'moving_threshold': 50.0,
            'min_threshold_pct': 10.0,
            'max_threshold_pct': 200.0,
            'threshold_step': 5.0
        },
        'static_min_threshold': {
            'min_value': 0.0,
            'min_range': -1000.0,
            'max_range': 1000.0,
            'step': 1.0
        },
        'static_max_threshold': {
            'max_value': 100.0,
            'min_range': 0.0,
            'max_range': 10000.0,
            'step': 1.0
        },
        'static_range_rule': {
            'lower_bound': 0.0,
            'upper_bound': 100.0,
            'min_range': -1000.0,
            'max_range': 1000.0,
            'step': 1.0
        },
        'static_percentage_rule': {
            'max_percentage': 20.0,
            'min_percentage': 1.0,
            'max_percentage_limit': 100.0,
            'step': 1.0,
            'baseline_value': 0.0
        },
        'rate_of_change': {
            'rate_threshold': 10.0,
            'min_rate': 1.0,
            'max_rate': 100.0,
            'step': 1.0,
            'window_size': 2
        },
        'consecutive_anomaly': {
            'consecutive_count': 3,
            'min_count': 2,
            'max_count': 10,
            'step': 1,
            'base_threshold': 2.0
        },
        'seasonal_decomp_rule': {
            'seasonal_period': 24,
            'min_period': 4,
            'max_period': 168,
            'step': 1,
            'seasonal_threshold': 2.0
        },
        'periodic_rule': {
            'period': 24,
            'min_period': 4,
            'max_period': 168,
            'step': 1,
            'threshold': 2.0,
            'periodic_rule_type':'shift_detection',
            'shift_threshold':0.3,
            'min_correlation':0.7,
            'window_size':192,
            'frequency_threshold':2.0,
            'num_periods':4
        },
        'percentile_rule': {
            'lower_percentile': 5.0,
            'upper_percentile': 95.0,
            'min_percentile': 1.0,
            'max_percentile': 99.0,
            'step': 0.5
        }
    }


    @staticmethod
    def detect_sharp_jumps(df, jump_threshold=2.0, direction='both'):
        """Detect sharp jumps in the data"""
        diff = df['value'].diff()
        mad = diff.abs().median()
        if direction == 'both':
            mask = diff.abs() > jump_threshold * mad
        elif direction == 'up':
            mask = diff > jump_threshold * mad
        elif direction == 'down':
            mask = diff < -jump_threshold * mad
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'both', 'up', or 'down'.")
        return mask

    @staticmethod
    def detect_baseline_deviations(df, baseline_window=50, deviation_threshold=2.0, baseline_method='mean'):
        # Compute rolling baseline per "hour" group
        if baseline_method == 'mean':
            baseline_value = df.groupby("hour")["value"].transform(lambda x: x.rolling(baseline_window, min_periods=1).mean())
        elif baseline_method == 'median':
            baseline_value = df.groupby("hour")["value"].transform(lambda x: x.rolling(baseline_window, min_periods=1).median())
        else:
            raise ValueError(f"Unsupported baseline_method: {baseline_method}")

        # Calculate % deviation from baseline
        baseline_deviation = np.abs(df['value'] - baseline_value) / baseline_value * 100

        # Return boolean mask where deviation exceeds threshold
        return baseline_deviation > deviation_threshold

    @staticmethod
    def detect_trend_anomalies(df, window=10, trend_threshold=None, trend_type='both'):
        """
        Detect anomalies in trend patterns.
    
        Parameters:
            df : DataFrame with 'value' column
            window : rolling window size
            trend_threshold : float or None, threshold for change in slope
            trend_type : 'both', 'up', or 'down'
        """
        # Calculate slope for each rolling window
        trends = df['value'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
       
        # Change in slope from one window to the next
        trend_changes = trends.diff()

        # Determine threshold
        if trend_threshold is None:
            # Use 95th percentile of absolute changes if no threshold provided
            threshold = trend_changes.abs().quantile(0.95)
        else:
            threshold = trend_threshold

        # Apply direction filtering
        if trend_type == 'both':
            mask = trend_changes.abs() > threshold
        elif trend_type == 'increasing':
            mask = trend_changes > threshold
        elif trend_type == 'decreasing':
            mask = trend_changes < -threshold
        else:
            raise ValueError(f"Invalid trend_type: {trend_type}")

        return mask

    @staticmethod
    def detect_periodic_shift(df, period, threshold):
        anomalies = np.zeros(len(df), dtype=bool)
        
        for i in range(period * 2, len(df)):
            current_period = df['value'].iloc[i-period:i]
            previous_period = df['value'].iloc[i-2*period:i-period]
            
            correlation = np.corrcoef(current_period, previous_period)[0, 1]
            
            if np.isnan(correlation) or correlation < (1 - threshold):
                anomalies[i-period:i] = True
        
        return anomalies


    @staticmethod 
    def detect_missing_periodicity(df, period, min_correlation, window_size):
        anomalies = np.zeros(len(df), dtype=bool)
        
        for i in range(window_size, len(df), period):
            window_data = df['value'].iloc[i-window_size:i]
            
            autocorr_at_period = np.corrcoef(
                window_data[:-period], 
                window_data[period:]
            )[0, 1] if len(window_data) > period else 0
            
            if np.isnan(autocorr_at_period) or autocorr_at_period < min_correlation:
                end_idx = min(i + period, len(df))
                anomalies[i:end_idx] = True
        
        return anomalies

    @staticmethod
    def detect_frequency_anomalies(df, period, threshold, window_size):
        from scipy.fft import fft
        
        anomalies = np.zeros(len(df), dtype=bool)
        
        if len(df) > window_size * 2:
            baseline_data = df['value'].iloc[:window_size]
            baseline_fft = np.abs(fft(baseline_data))
            
            for i in range(window_size, len(df), period):
                end_idx = min(i + window_size, len(df))
                current_data = df['value'].iloc[i:end_idx]
                
                if len(current_data) == window_size:
                    current_fft = np.abs(fft(current_data))
                    
                    freq_diff = np.mean(np.abs(current_fft - baseline_fft))
                    freq_baseline = np.mean(baseline_fft)
                    
                    if freq_diff / (freq_baseline + 1e-8) > threshold:
                        anomalies[i:end_idx] = True
        
        return anomalies


    @staticmethod
    def apply_rule_based_detection(df, rule_methods, all_rules_config):
        """Apply rule-based detection methods"""
        results = {}
    
        print(f"apply_rule_based_detection(): rule_methods are : {rule_methods} and all_rules_config is : {all_rules_config}")
        for rule_name, rule_info in rule_methods.items():
            if not rule_info['enabled']:
                print(f"apply_rule_based_detection(): rule_method : {rule_name} not enabled : rule_info is : {rule_info['enabled']} ")
                continue

            print(f" rule name in apply_rule_based_detection() is : {rule_name}") 
            rule_config = all_rules_config.get(rule_name, {})
            
            if rule_name == 'sharp_jump_rule':
                results[rule_name] = RuleBasedAnomaly.detect_sharp_jumps( df, rule_config.get('jump_threshold', 2.0), direction=rule_config.get('direction', 'both'))
            
            elif rule_name == 'trend_rule':
                results[rule_name] = RuleBasedAnomaly.detect_trend_anomalies( df, rule_config.get('trend_window', 10), trend_threshold=rule_config.get('trend_threshold', 0.8), trend_type=rule_config.get('trend_type', 'both')
                )
            
            elif rule_name == 'baseline_deviation_rule':
                results[rule_name] = RuleBasedAnomaly.detect_baseline_deviations( df, rule_config.get('baseline_window', 50), deviation_threshold=rule_config.get('deviation_threshold', 2.0), baseline_method=rule_config.get('baseline_method', 'mean')
                )
            
            elif rule_name == 'zscore_threshold':
                threshold = rule_config.get('zscore_value', 3.0)
                use_modified = rule_config.get('use_modified', False)
                if use_modified:
                    median = df['value'].median()
                    mad = np.median(np.abs(df['value'] - median))
                    modified_z_scores = 0.6745 * (df['value'] - median) / mad
                    results[rule_name] = np.abs(modified_z_scores) > threshold
                else:
                    z_scores = np.abs(stats.zscore(df['value']))
                    results[rule_name] = z_scores > threshold
                
            elif rule_name == 'iqr_method':
                multiplier = rule_config.get('iqr_multiplier', 1.5)
                use_median = rule_config.get('use_median', False)
            
                if use_median:
                    median_val = df['value'].median()
                    Q1 = df['value'].quantile(0.25)
                    Q3 = df['value'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = median_val - multiplier * IQR
                    upper_bound = median_val + multiplier * IQR
                else:
                    Q1 = df['value'].quantile(0.25)
                    Q3 = df['value'].quantile(0.75) 
                    IQR = Q3 - Q1
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                
                results[rule_name] = (df['value'] < lower_bound) | (df['value'] > upper_bound)
            
            elif rule_name == 'moving_avg':
                window = rule_config.get('moving_window', 10)
                threshold_pct = rule_config.get('moving_threshold', 50.0)
                center_window = rule_config.get('center_window', True)
            
                df_temp = df.copy()
                df_temp['moving_avg'] = df_temp['value'].rolling(
                    window=window, 
                    center=center_window
                ).mean()
                df_temp['deviation'] = np.abs(df_temp['value'] - df_temp['moving_avg']) / df_temp['moving_avg'] * 100
                results[rule_name] = df_temp['deviation'] > threshold_pct
            
            # Data is anomaly if value < min_threshold (below minimum acceptable value)
            elif rule_name == 'static_min_threshold':
                min_val = rule_config.get('min_value', 0.0)
                inclusive = rule_config.get('inclusive', True)
            
                if inclusive:
                    results[rule_name] = df['value'] <= min_val  # ≤ min_val = anomaly
                else:
                    results[rule_name] = df['value'] < min_val   # < min_val = anomaly
            
                print(f"Static Min Threshold: {results[rule_name].sum()} values below {min_val}")
 

            elif rule_name == 'static_max_threshold':
                max_val = rule_config.get('max_value', 100.0)
                print(f" max_val is : {max_val}")
                inclusive = rule_config.get('inclusive', True)
                if inclusive:
                    results[rule_name] = df['value'] >= max_val
                else:
                    results[rule_name] = df['value'] > max_val

                print(f"Static Max Threshold: {results[rule_name].sum()} values above {max_val}")    

            elif rule_name == 'static_range_rule':
                lower = rule_config.get('lower_bound', 0.0)
                upper = rule_config.get('upper_bound', 100.0)
                inclusive = rule_config.get('inclusive', True)
                if inclusive:
                    results[rule_name] = (df['value'] < lower) | (df['value'] > upper)
                else:
                    results[rule_name] = (df['value'] <= lower) | (df['value'] >= upper)
                
            # Data is anomaly if value exceeds a percentage of the maximum range
            elif rule_name == 'static_percentage_rule':
                threshold_percentage = rule_config.get('threshold_percentage', 94.0)  # 94%
                min_value = rule_config.get('min_value', 0.0)  # Default min = 0
                max_value = rule_config.get('max_value', 100.0)  # Default max = 100
            
                # Calculate the threshold value (percentage of max range)
                range_size = max_value - min_value
                threshold_value = min_value + (range_size * threshold_percentage / 100.0)
            
                # Data is anomaly if it exceeds the threshold percentage of range
                results[rule_name] = df['value'] >= threshold_value
            
                print(f"Static Percentage Rule: {results[rule_name].sum()} values ≥ {threshold_value:.2f} (which is {threshold_percentage}% of range [{min_value}, {max_value}])")

            # RATE OF CHANGE RULE
            # Data is anomaly if rate of change between consecutive windows > threshold
            elif rule_name == 'rate_of_change':
                rate_threshold = rule_config.get('rate_threshold', 10.0)
                window_size = rule_config.get('window_size', 1)  # Default: point-to-point
            
                # Calculate rate of change
                if window_size == 1:
                    # Point-to-point rate of change
                    rate_of_change = df['value'].diff().abs()
                else:
                    # Rate of change over window
                    rate_of_change = df['value'].diff(periods=window_size).abs() / window_size
            
                results[rule_name] = rate_of_change > rate_threshold
            
                print(f"Rate of Change Rule: {results[rule_name].sum()} values with rate > {rate_threshold}")

        # CONSECUTIVE ANOMALY RULE
        # Data is anomaly if it's part of N consecutive anomalous points
            elif rule_name == 'consecutive_anomaly':
                consecutive_count = rule_config.get('consecutive_count', 3)
                base_threshold = rule_config.get('base_threshold', 2.0)
                base_method = rule_config.get('base_method', 'zscore')  # Method to detect individual anomalies
            
                # Step 1: Detect individual anomalies using base method
                if base_method == 'zscore':
                    z_scores = np.abs(stats.zscore(df['value']))
                    individual_anomalies = z_scores > base_threshold
                elif base_method == 'iqr':
                    Q1 = df['value'].quantile(0.25)
                    Q3 = df['value'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - base_threshold * IQR
                    upper = Q3 + base_threshold * IQR
                    individual_anomalies = (df['value'] < lower) | (df['value'] > upper)
                else:  # deviation from mean
                    mean_val = df['value'].mean()
                    std_val = df['value'].std()
                    threshold_val = base_threshold * std_val
                    individual_anomalies = np.abs(df['value'] - mean_val) > threshold_val
            
                # Step 2: Find consecutive anomalies
                consecutive_anomalies = np.zeros(len(df), dtype=bool)
                current_streak = 0
                streak_start = 0
            
                for i, is_anomaly in enumerate(individual_anomalies):
                    if is_anomaly:
                        if current_streak == 0:
                            streak_start = i
                        current_streak += 1
                    
                        # If we hit the required consecutive count
                        if current_streak >= consecutive_count:
                            # Mark all points in this consecutive sequence as anomalies
                            consecutive_anomalies[streak_start:i+1] = True
                    else:
                        current_streak = 0
            
                results[rule_name] = consecutive_anomalies
            
                print(f"Consecutive Anomaly Rule: {results[rule_name].sum()} points in consecutive sequences of ≥{consecutive_count}")

            
            elif rule_name == 'seasonal_decomp_rule':
                period = rule_config.get('seasonal_period', 24)
                threshold = rule_config.get('seasonal_threshold', 2.0)
            
                # Calculate seasonal decomposition (simplified)
                seasonal_mean = df.groupby(df.index % period)['value'].transform('mean')
                seasonal_std = df.groupby(df.index % period)['value'].transform('std')
                seasonal_deviation = np.abs(df['value'] - seasonal_mean) / seasonal_std
                results[rule_name] = seasonal_deviation > threshold

            elif rule_name == 'periodic_rule':
                period = rule_config.get('period', 24)
                periodic_rule_type = rule_config.get('periodic_rule_type', 'shift_detection')
    
                if periodic_rule_type == 'shift_detection':
                    threshold = rule_config.get('shift_threshold', 0.3)
                    results[rule_name] = RuleBasedAnomaly.detect_periodic_shift(df, period, threshold)
        
                elif periodic_rule_type == 'missing_periodicity':
                    min_correlation = rule_config.get('min_correlation', 0.7)
                    window_size = rule_config.get('window_size', period * 4)
                    results[rule_name] = RuleBasedAnomaly.detect_missing_periodicity(df, period, min_correlation, window_size)
        
                elif periodic_rule_type == 'frequency_domain':
                    threshold = rule_config.get('frequency_threshold', 2.0)
                    window_size = rule_config.get('window_size', period * 4)
                    results[rule_name] = RuleBasedAnomaly.detect_frequency_anomalies(df, period, threshold, window_size)

            
            elif rule_name == 'percentile_rule':
                # What it detects: Statistical outliers based on the data's own distribution
                # How it works: Calculates percentiles from the actual data and flags values outside those bounds
                # Use case: Detecting unusual behavior relative to historical patterns

                lower_pct = rule_config.get('lower_percentile', 5.0)
                upper_pct = rule_config.get('upper_percentile', 95.0)
            
                lower_bound = df['value'].quantile(lower_pct / 100)
                upper_bound = df['value'].quantile(upper_pct / 100)
                results[rule_name] = (df['value'] < lower_bound) | (df['value'] > upper_bound)
    
        return results
