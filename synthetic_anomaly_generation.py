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
from preprocess_analyze_properties import get_tooltip, analyze_data_properties, select_recommended_methods, ensure_boolean_anomaly_column
from rule_based_methods import RuleBasedAnomaly
from ml_based_models import MLBasedAnomaly
from evaluate_metrics import calculate_metrics, calculate_detailed_metrics

    
def get_holidays(start_date, end_date, country='US'):
    """Get holidays for the date range with error handling"""
    try:
        country_holidays = holidays.country_holidays(country)
        holiday_list = []
        current = start_date
        while current <= end_date:
            if current in country_holidays:
                holiday_list.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'name': country_holidays[current]
                })
            current += timedelta(days=1)
        return holiday_list
    except Exception as e:
        st.warning(f"Could not load holidays: {e}. Using default holidays.")
        # Fallback to common holidays
        return [
            {'date': '2024-01-01', 'name': 'New Year'},
            {'date': '2024-07-04', 'name': 'Independence Day'},
            {'date': '2024-12-25', 'name': 'Christmas'},
            {'date': '2024-11-28', 'name': 'Thanksgiving'}
    ]
    
def calculate_time_features(timestamp, peak_hours, off_hours):
    """Calculate comprehensive time-based features with detailed explanations"""
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    hour = timestamp.hour
    is_weekend = day_of_week >= 5
    is_peak_hour = hour in peak_hours
    is_off_hour = hour in off_hours
        
    return {
        'day_of_week': day_of_week,
        'hour': hour,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'is_off_hour': is_off_hour,
        'hour_sin': np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),  # Cyclical day encoding
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        'week_of_year': timestamp.isocalendar()[1],
        'day_of_month': timestamp.day,
        'month': timestamp.month
    }
    
def generate_synthetic_data(config):
    """Generate synthetic time series data with comprehensive configurability"""
    data = []
    current_time = config['start_date']
    holiday_list = [h['date'] for h in get_holidays(config['start_date'], config['end_date'])]
    
    interval = timedelta(minutes=config['interval_minutes'])
    total_points = int((config['end_date'] - config['start_date']).total_seconds() / (config['interval_minutes'] * 60))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    i = 0
    while current_time <= config['end_date']:
        time_features = calculate_time_features(current_time, config['peak_hours'], config['off_hours'])
        date_str = current_time.strftime('%Y-%m-%d')
        is_holiday = date_str in holiday_list
        
        # Base value - starting point for all calculations
        base_value = config['base_value']
        
        # SEASONALITY: Regular patterns that repeat over time
        seasonal_component = 0
        
        # Daily seasonality: Values change predictably throughout the day
        if config['daily_seasonality']:
            daily_strength = config['seasonality_strength']['daily']
            seasonal_component += np.sin(2 * np.pi * time_features['hour'] / 24) * daily_strength
        
        # Weekly seasonality: Values change predictably throughout the week
        if config['weekly_seasonality']:
            weekly_strength = config['seasonality_strength']['weekly']
            seasonal_component += np.sin(2 * np.pi * time_features['day_of_week'] / 7) * weekly_strength
        
        # Monthly seasonality: Values change predictably throughout the month
        if config['monthly_seasonality']:
            monthly_strength = config['seasonality_strength']['monthly']
            seasonal_component += np.sin(2 * np.pi * time_features['day_of_month'] / 30) * monthly_strength
        
        # TREND MULTIPLIERS: How different time contexts affect the base value
        # These multiply the base+seasonal value to simulate real-world patterns
        trend_multiplier = 1.0
        
        if is_holiday:
            # Holiday trend: How values change on holidays (usually lower for business metrics)
            trend_multiplier = config['trends']['holiday']
        elif time_features['is_weekend']:
            # Weekend trend: How values change on weekends
            trend_multiplier = config['trends']['weekend']
        elif time_features['is_peak_hour']:
            # Peak hour trend: How values change during busy periods
            trend_multiplier = config['trends']['peak_hour']
        elif time_features['is_off_hour']:
            # Off hour trend: How values change during quiet periods
            trend_multiplier = config['trends']['off_hour']
        else:
            # Weekday-specific trends: Different behavior for each day of the week
            weekday_trends = config['trends']['weekdays']
            trend_multiplier = weekday_trends[time_features['day_of_week']]
        
        # Calculate final value: (base + seasonality) × trend_multiplier
        value = (base_value + seasonal_component) * trend_multiplier
        
        # Add controlled noise: Random variation to make data realistic
        noise = np.random.normal(0, config['noise_level'])
        value += noise
        
        # Ensure non-negative values
        value = max(0, value)
        
        # Create comprehensive data point
        data_point = {
            'timestamp': current_time,
            'value': value,
            'is_anomaly': False,
            'anomaly_type': 'normal',
            'manual_label': 'normal',  # For manual editing
            'confidence': 1.0,  # Confidence in the label
            'is_weekend': time_features['is_weekend'],
            'is_peak_hour': time_features['is_peak_hour'],
            'is_off_hour': time_features['is_off_hour'],
            'is_holiday': is_holiday,
            'hour': time_features['hour'],
            'day_of_week': time_features['day_of_week'],
            'week_of_year': time_features['week_of_year'],
            'month': time_features['month'],
            'day_of_month': time_features['day_of_month'],
            'hour_sin': time_features['hour_sin'],
            'hour_cos': time_features['hour_cos'],
            'day_sin': time_features['day_sin'],
            'day_cos': time_features['day_cos']
        }
            
        data.append(data_point)
        current_time += interval
        i += 1
        
        # Update progress every 100 points
        if i % 100 == 0:
            progress = min(i / total_points, 1.0)
            progress_bar.progress(progress)
            status_text.text(f'Generating data... {i:,}/{total_points:,} points')
    
    progress_bar.progress(1.0)
    status_text.text(f'✅ Generated {len(data):,} data points successfully!')
    
    df = pd.DataFrame(data)
    
    # Add synthetic anomalies
    # df = add_synthetic_anomalies(df, config)
    df, total_anomaly_instances, total_anomalous_points = generate_exclusive_anomalies(df, config)
        
    return df, total_anomaly_instances, total_anomalous_points


from collections import defaultdict

def generate_exclusive_anomalies(df, config):
    """
    Generate anomalies with strict exclusivity and priority resolution
    Priority: Pattern > Contextual > Point
    """

    # Initialize tracking
    total_anomaly_instances = 0
    total_anomalous_points = 0
    occupied_indices = set()  # Track indices already used for anomalies

    # Initialize anomaly tracking columns
    df['is_anomaly'] = False
    df['anomaly_type'] = None
    df['anomaly_id'] = None
    df['confidence'] = 0.0
    df['pattern_group_id'] = None  # For pattern anomalies
    df['pattern_position'] = None  # Position within pattern (0, 1, 2, ...)

    print("=== EXCLUSIVE ANOMALY GENERATION ===")

    # STEP 1: Generate Pattern Anomalies (Highest Priority)
    if config['anomalies']['pattern']['enabled']:
        rate = config['anomalies']['pattern']['rate']
        n_pattern_points = int(len(df) * rate / 100)
        print(f"DEBUG INFO: Pattern anomalies rate: {rate}%, Target points: {n_pattern_points} out of {len(df)}")
    
        points_created = 0
        pattern_instance = 0
        attempts = 0
        max_attempts = 1000  # Prevent infinite loops
    
        while points_created < n_pattern_points and attempts < max_attempts:
            attempts += 1
        
            # Find a suitable location for pattern
            start_idx = np.random.randint(10, len(df) - 10)
            pattern_length = np.random.randint(3, 8)
            end_idx = min(start_idx + pattern_length, len(df))
            actual_length = end_idx - start_idx
        
            # Check if this range is available
            pattern_range = set(range(start_idx, end_idx))
            if pattern_range.intersection(occupied_indices):
                continue  # Skip if overlap found
        
            # Adjust length to not exceed remaining points needed
            actual_length = min(actual_length, n_pattern_points - points_created)
            end_idx = start_idx + actual_length
        
            # Create the pattern anomaly
            for j, idx in enumerate(range(start_idx, end_idx)):
                if np.random.random() < 0.5:
                    # Gradual drift
                    drift_factor = config['anomalies']['pattern']['intensity'] * (j / actual_length)
                    df.at[idx, 'value'] *= (1 + drift_factor)
                else:
                    # Sudden change
                    df.at[idx, 'value'] *= config['anomalies']['pattern']['intensity']
            
                # Mark as pattern anomaly
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'pattern'
                df.at[idx, 'anomaly_id'] = f'pattern_{pattern_instance}'
                df.at[idx, 'pattern_group_id'] = pattern_instance
                df.at[idx, 'pattern_position'] = j
                df.at[idx, 'confidence'] = 0.7
            
                occupied_indices.add(idx)
                total_anomalous_points += 1
        
            total_anomaly_instances += 1
            pattern_instance += 1
            points_created += actual_length
        
            if points_created >= n_pattern_points:
               break
    
        print(f"Pattern anomalies created: {total_anomaly_instances} instances, {points_created} points")

    # STEP 2: Generate Contextual Anomalies (Medium Priority)
    if config['anomalies']['contextual']['enabled']:
        rate = config['anomalies']['contextual']['rate']
        n_contextual = int(len(df) * rate / 100)
        print(f"DEBUG INFO: Contextual anomalies rate: {rate}%, Target points: {n_contextual} out of {len(df)}")
    
        # Get available indices for each time context
        peak_indices = df[df['is_peak_hour']].index.tolist()
        off_indices = df[df['is_off_hour']].index.tolist()
    
        # Remove already occupied indices
        available_peak = [idx for idx in peak_indices if idx not in occupied_indices]
        available_off = [idx for idx in off_indices if idx not in occupied_indices]
    
        contextual_created = 0
    
        # Distribute contextual anomalies
        peak_anomalies = n_contextual // 2
        off_anomalies = n_contextual - peak_anomalies
    
        # Peak hour anomalies (unexpectedly low values)
        if peak_anomalies > 0 and len(available_peak) > 0:
            actual_peak = min(peak_anomalies, len(available_peak))
            selected_peak = np.random.choice(available_peak, size=actual_peak, replace=False)
        
            for idx in selected_peak:
                df.at[idx, 'value'] *= config['anomalies']['contextual']['intensity'] * 0.3
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'contextual'
                df.at[idx, 'anomaly_id'] = f'contextual_peak_{contextual_created}'
                df.at[idx, 'confidence'] = 0.8
            
                occupied_indices.add(idx)
                total_anomaly_instances += 1
                total_anomalous_points += 1
                contextual_created += 1
    
        # Off hour anomalies (unexpectedly high values)
        if off_anomalies > 0 and len(available_off) > 0:
            actual_off = min(off_anomalies, len(available_off))
            selected_off = np.random.choice(available_off, size=actual_off, replace=False)
        
            for idx in selected_off:
                df.at[idx, 'value'] *= config['anomalies']['contextual']['intensity'] * 2.5
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'contextual'
                df.at[idx, 'anomaly_id'] = f'contextual_off_{contextual_created}'
                df.at[idx, 'confidence'] = 0.8
            
                occupied_indices.add(idx)
                total_anomaly_instances += 1
                total_anomalous_points += 1
                contextual_created += 1
    
        print(f"Contextual anomalies created: {contextual_created} points")

    # STEP 3: Generate Point Anomalies (Lowest Priority)
    if config['anomalies']['point']['enabled']:
        rate = config['anomalies']['point']['rate']
        n_point = int(len(df) * rate / 100)
        print(f"DEBUG INFO: Point anomalies rate: {rate}%, Target points: {n_point} out of {len(df)}")
    
        # Get available indices (not occupied by pattern or contextual anomalies)
        available_indices = [idx for idx in df.index if idx not in occupied_indices]
    
        if len(available_indices) > 0:
            actual_point = min(n_point, len(available_indices))
            selected_indices = np.random.choice(available_indices, size=actual_point, replace=False)
        
            point_created = 0
            for idx in selected_indices:
                df.at[idx, 'value'] *= config['anomalies']['point']['intensity']
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'point'
                df.at[idx, 'anomaly_id'] = f'point_{point_created}'
                df.at[idx, 'confidence'] = 0.9
            
                occupied_indices.add(idx)
                total_anomaly_instances += 1
                total_anomalous_points += 1
                point_created += 1
        
            print(f"Point anomalies created: {point_created} points")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total Anomaly Instances: {total_anomaly_instances}")
    print(f"Total Anomalous Points: {total_anomalous_points}")
    print(f"Occupied Indices: {len(occupied_indices)}")
    print(f"Anomalous Percentage: {total_anomalous_points/len(df)*100:.2f}%")

    # Verification
    verify_exclusivity(df)

    return df, total_anomaly_instances, total_anomalous_points

def verify_exclusivity(df):
    """Verify that no anomalies overlap"""
    anomaly_counts = df['anomaly_type'].value_counts()
    total_marked = df['is_anomaly'].sum()

    print(f"\n=== EXCLUSIVITY VERIFICATION ===")
    print(f"Pattern anomalies: {anomaly_counts.get('pattern', 0)}")
    print(f"Contextual anomalies: {anomaly_counts.get('contextual', 0)}")
    print(f"Point anomalies: {anomaly_counts.get('point', 0)}")
    print(f"Total marked as anomaly: {total_marked}")
    print(f"Sum of individual types: {anomaly_counts.sum()}")

    if total_marked == anomaly_counts.sum():
        print("✅ EXCLUSIVITY VERIFIED: No overlaps detected")
    else:
        print("❌ EXCLUSIVITY FAILED: Overlaps detected!")


def get_all_anomalous_timestamps(generated_anomalies):
    all_timestamps = []
    
    # 1. Handle Contextual Anomalies (list of single indices)
    if 'contextual' in generated_anomalies and 'timestamps' in generated_anomalies['contextual']:
        all_timestamps.extend(generated_anomalies['contextual']['timestamps'])
    
    # 2. Handle Point Anomalies (list of single indices)
    if 'point' in generated_anomalies and 'timestamps' in generated_anomalies['point']:
        all_timestamps.extend(generated_anomalies['point']['timestamps'])
    
    # 3. Handle Pattern Anomalies (list of [start, end] pairs)
    if 'pattern' in generated_anomalies and 'timestamps' in generated_anomalies['pattern']:
        for start, end in generated_anomalies['pattern']['timestamps']:
            # Create a list of all indices within the pattern's range
            all_timestamps.extend(list(range(start, end + 1)))
            
    return all_timestamps


def evaluate_with_strict_rules(df, predictions, pattern_detection_threshold=0.6):
    """
    Evaluate detection results with STRICT rules:
    - Pattern anomalies: Count as TP only if >= threshold% consecutive detection
    - Individual detections within patterns that don't meet threshold = FP
    - Contextual/Point anomalies: Standard point-level evaluation
    """

    # Convert predictions to binary if needed
    predictions = np.array(predictions).astype(int)
    
    # Get detailed evaluations
    pattern_eval = evaluate_pattern_detection_strict(df, predictions, pattern_detection_threshold)
    point_eval = evaluate_point_detection(df, predictions)

    # Calculate STRICT overall metrics
    strict_metrics = calculate_strict_overall_metrics(df, predictions, pattern_eval, point_eval)
    
    results = {
        'pattern_evaluation': pattern_eval,
        'point_evaluation': point_eval,
        'strict_overall_metrics': strict_metrics
    }
    
    return results

def evaluate_pattern_detection_strict(df, predictions, threshold=0.6):
    """
    STRICT evaluation for pattern anomalies:
    - Pattern counts as TP only if consecutive detection >= threshold
    - Individual detections within failed patterns count as FP
    """
    pattern_df = df[df['anomaly_type'] == 'pattern']
    if len(pattern_df) == 0:
        return {
            'total_patterns': 0, 'detected_patterns': 0, 'pattern_detection_rate': 0.0,
            'pattern_details': [], 'valid_tp_indices': [], 'invalid_fp_indices': []
        }

    # Group by pattern
    pattern_groups = pattern_df.groupby('pattern_group_id')
    total_patterns = len(pattern_groups)
    detected_patterns = 0

    pattern_details = []
    valid_tp_indices = []  # Indices that count as TP (from successful patterns)
    invalid_fp_indices = []  # Indices that count as FP (from failed patterns)

    for pattern_id, group in pattern_groups:
        indices = group.index.tolist()
        pattern_predictions = predictions[indices]
    
        # Check for consecutive detection >= threshold
        total_points = len(indices)
        detected_points = np.sum(pattern_predictions)
        detection_rate = detected_points / total_points
    
        # Find longest consecutive detection sequence
        consecutive_detections = find_longest_consecutive(pattern_predictions)
        consecutive_rate = consecutive_detections / total_points
    
        # STRICT RULE: Pattern is detected only if consecutive detection >= threshold
        is_detected = consecutive_rate >= threshold
    
        if is_detected:
            detected_patterns += 1
            # All detected points in successful pattern count as TP
            for i, idx in enumerate(indices):
                if pattern_predictions[i] == 1:
                    valid_tp_indices.append(idx)
        else:
            # All detected points in failed pattern count as FP
            for i, idx in enumerate(indices):
                if pattern_predictions[i] == 1:
                    invalid_fp_indices.append(idx)
    
        pattern_details.append({
            'pattern_id': pattern_id,
            'indices': indices,
            'total_points': total_points,
            'detected_points': detected_points,
            'detection_rate': detection_rate,
            'consecutive_detections': consecutive_detections,
            'consecutive_rate': consecutive_rate,
            'is_detected': is_detected,
            'status': 'SUCCESS' if is_detected else 'FAILED'
        })
    
    return {
        'total_patterns': total_patterns,
        'detected_patterns': detected_patterns,
        'pattern_detection_rate': detected_patterns / total_patterns if total_patterns > 0 else 0.0,
        'pattern_details': pattern_details,
        'valid_tp_indices': valid_tp_indices,
        'invalid_fp_indices': invalid_fp_indices
    }

def find_longest_consecutive(binary_array):
    """Find the length of longest consecutive 1s in binary array"""
    if len(binary_array) == 0:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for val in binary_array:
        if val == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive

def evaluate_point_detection(df, predictions):
    """
    Evaluate contextual and point anomalies (standard point-level evaluation)
    """
    point_types = ['contextual', 'point']
    point_df = df[df['anomaly_type'].isin(point_types)]

    if len(point_df) == 0:
        return {'total_points': 0, 'detected_points': 0, 'detection_rate': 0.0}

    indices = point_df.index.tolist()
    ground_truth = np.ones(len(indices))  # All are anomalies
    point_predictions = predictions[indices]

    detected_points = np.sum(point_predictions)
    total_points = len(indices)

    return {
        'total_points': total_points,
        'detected_points': detected_points,
        'detection_rate': detected_points / total_points if total_points > 0 else 0.0
    }

def calculate_strict_overall_metrics(df, predictions, pattern_eval, point_eval):
    """
    Calculate STRICT overall metrics following the rules:
    - Only successful patterns contribute to TP
    - Failed pattern detections count as FP
    - Contextual/Point anomalies use standard evaluation
    """

    # Get all indices
    all_indices = set(df.index)

    # Pattern-related indices
    pattern_valid_tp = set(pattern_eval['valid_tp_indices'])
    pattern_invalid_fp = set(pattern_eval['invalid_fp_indices'])

    # Point-level anomaly indices (contextual + point)
    point_types = ['contextual', 'point']
    point_anomaly_indices = set(df[df['anomaly_type'].isin(point_types)].index)

    # Normal (non-anomalous) indices
    normal_indices = set(df[~df['is_anomaly']].index)

    # Calculate TP, FP, FN, TN
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    tp_details = []
    fp_details = []
    fn_details = []

    for idx in all_indices:
        predicted = predictions[idx]
    
        if idx in pattern_valid_tp:
            # Index is part of a successful pattern
            if predicted == 1:
                tp += 1
                tp_details.append(f"Pattern point {idx} (successful pattern)")
            else:
                fn += 1
                fn_details.append(f"Pattern point {idx} (missed in successful pattern)")
            
        elif idx in pattern_invalid_fp:
            # Index is part of a failed pattern
            if predicted == 1:
                fp += 1
                fp_details.append(f"Pattern point {idx} (failed pattern)")
            # If predicted == 0, it doesn't count as anything (pattern failed anyway)
        
        elif idx in point_anomaly_indices:
            # Index is contextual or point anomaly
            if predicted == 1:
                tp += 1
                anomaly_type = df.at[idx, 'anomaly_type']
                tp_details.append(f"{anomaly_type.title()} anomaly {idx}")
            else:
                fn += 1
                anomaly_type = df.at[idx, 'anomaly_type']
                fn_details.append(f"{anomaly_type.title()} anomaly {idx}")
            
        elif idx in normal_indices:
            # Index is normal (not anomalous)
            if predicted == 1:
                fp += 1
                fp_details.append(f"Normal point {idx} (false alarm)")
            else:
                tn += 1
    
        else:
            # Index is part of a failed pattern but not detected (no penalty, no reward)
            if predicted == 0:
                # This is expected for failed patterns
                pass

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'tp_details': tp_details,
        'fp_details': fp_details,
        'fn_details': fn_details
    }

def strict_evaluation_report(df, predictions, pattern_threshold=0.6):
    """
    Generate STRICT evaluation report following the rules:
    - Pattern anomalies must have >= threshold consecutive detection to count as TP
    - Failed pattern detections count as FP
    - Individual anomalies use standard point-level evaluation
    """
    results = evaluate_with_strict_rules(df, predictions, pattern_threshold)

    print("=== STRICT ANOMALY DETECTION EVALUATION ===")
    print(f"Pattern Success Threshold: {pattern_threshold*100}% consecutive detection required")
    print("Rule: Pattern detections only count as TP if pattern succeeds, otherwise FP")
    print()

    # Overall STRICT metrics
    strict = results['strict_overall_metrics']
    print("STRICT OVERALL METRICS:")
    print(f"  Precision: {strict['precision']:.3f}")
    print(f"  Recall: {strict['recall']:.3f}")
    print(f"  F1-Score: {strict['f1_score']:.3f}")
    print(f"  Accuracy: {strict['accuracy']:.3f}")

    confusion = strict['confusion_matrix']
    print(f"  Confusion Matrix: TP={confusion['tp']}, FP={confusion['fp']}, "
          f"FN={confusion['fn']}, TN={confusion['tn']}")
    print()

    # Pattern-specific analysis
    pattern = results['pattern_evaluation']
    if pattern['total_patterns'] > 0:
        print("PATTERN ANOMALY ANALYSIS:")
        print(f"  Total Pattern Instances: {pattern['total_patterns']}")
        print(f"  Successfully Detected Patterns: {pattern['detected_patterns']}")
        print(f"  Pattern Success Rate: {pattern['pattern_detection_rate']:.3f}")
        print(f"  Valid TP from Patterns: {len(pattern['valid_tp_indices'])}")
        print(f"  Invalid FP from Failed Patterns: {len(pattern['invalid_fp_indices'])}")
        print()
        
        # Detailed pattern breakdown
        print("PATTERN BREAKDOWN:")
        for detail in pattern['pattern_details']:
            status_icon = "✅" if detail['status'] == 'SUCCESS' else "❌"
            print(f"  {status_icon} Pattern {detail['pattern_id']} ({detail['status']})")
            print(f"     Indices: {detail['indices']}")
            print(f"     Detected: {detail['detected_points']}/{detail['total_points']} points "
                  f"({detail['detection_rate']:.1%})")
            print(f"     Consecutive: {detail['consecutive_detections']} points "
                  f"({detail['consecutive_rate']:.1%})")
        
            if detail['status'] == 'FAILED' and detail['detected_points'] > 0:
                print(f"     ⚠️  {detail['detected_points']} detections counted as FALSE POSITIVES")
        print()
    
    # Point-level analysis
    point = results['point_evaluation']
    if point['total_points'] > 0:
        print("INDIVIDUAL ANOMALY ANALYSIS:")
        print(f"  Total Individual Anomalies: {point['total_points']}")
        print(f"  Successfully Detected: {point['detected_points']}")
        print(f"  Individual Detection Rate: {point['detection_rate']:.3f}")
        print()
    
    # Detailed breakdown of TPs and FPs
    if len(strict['tp_details']) > 0:
        print("TRUE POSITIVES:")
        for detail in strict['tp_details']:
            print(f"  ✅ {detail}")
        print()
   
    if len(strict['fp_details']) > 0:
        print("FALSE POSITIVES:")
        for detail in strict['fp_details']:
            print(f"  ❌ {detail}")
        print()
  
    if len(strict['fn_details']) > 0:
        print("FALSE NEGATIVES:")
        for detail in strict['fn_details']:
            print(f"  ⭕ {detail}")
            print()
    
    return results

def demonstrate_strict_evaluation():
    """
    Demonstrate the strict evaluation with your example:
    - Pattern: X to X+5 (indices 10-15)
    - Contextual: Y, Z (indices 20, 25)  
    - Point: M (index 30)
    - Model detections: X, X+2, X+3, Z, M (indices 10, 12, 13, 25, 30)
    """
    print("=== DEMONSTRATION OF STRICT EVALUATION ===")
    print("Ground Truth:")
    print("  Pattern anomaly: indices 10-15 (6 points)")
    print("  Contextual anomalies: indices 20, 25")  
    print("  Point anomaly: index 30")
    print("  Total anomalous points: 9")
    print()
    print("Model Predictions:")
    print("  Detected indices: 10, 12, 13, 25, 30 (5 detections)")
    print()

    # Create sample dataframe for demonstration
    df_demo = pd.DataFrame({
        'value': np.random.randn(50),
        'is_anomaly': False,
        'anomaly_type': None,
        'pattern_group_id': None
    })

    # Set up anomalies
    # Pattern: 10-15
    for idx in range(10, 16):
        df_demo.at[idx, 'is_anomaly'] = True
        df_demo.at[idx, 'anomaly_type'] = 'pattern'
        df_demo.at[idx, 'pattern_group_id'] = 0

    # Contextual: 20, 25
    for idx in [20, 25]:
        df_demo.at[idx, 'is_anomaly'] = True
        df_demo.at[idx, 'anomaly_type'] = 'contextual'

    # Point: 30
    df_demo.at[30, 'is_anomaly'] = True
    df_demo.at[30, 'anomaly_type'] = 'point'

    # Model predictions: 10, 12, 13, 25, 30
    predictions_demo = np.zeros(50)
    predictions_demo[[10, 12, 13, 25, 30]] = 1

    # Evaluate with different thresholds
    print("EVALUATION WITH 60% THRESHOLD:")
    results_60 = strict_evaluation_report(df_demo, predictions_demo, 0.6)

    print("\nEVALUATION WITH 40% THRESHOLD:")
    results_40 = strict_evaluation_report(df_demo, predictions_demo, 0.4)

    return df_demo, predictions_demo, results_60, results_40


def add_synthetic_anomalies(df, config):
    """Add different types of synthetic anomalies with detailed explanations"""
    df = df.copy()
    total_anomalies = 0
    total_anomaly_events = 0
    
    # CONTEXTUAL ANOMALIES: Values that are normal in isolation but abnormal given the context
    # Example: Low traffic during rush hour, high traffic at 3 AM
    if config['anomalies']['contextual']['enabled']:
        rate = config['anomalies']['contextual']['rate']  # Percentage of data to make anomalous
        n_contextual = int(len(df) * rate / 100)
        print(f" DEBUG INFO : contextual anomalies rate : {rate} , Number is : {n_contextual} out of {len(df)}")
        
        # Strategy: Create context-inappropriate values
        peak_indices = df[df['is_peak_hour']].index.tolist()
        off_indices = df[df['is_off_hour']].index.tolist()

        # Fix for odd numbers
        num_peak_anomalies = n_contextual // 2
        num_off_anomalies = n_contextual - num_peak_anomalies
        
        if len(peak_indices) > 0:
            selected_peak = np.random.choice(peak_indices, size=min(num_peak_anomalies, len(peak_indices)), replace=False)
            for idx in selected_peak:
                # Make peak hour values unexpectedly low
                df.at[idx, 'value'] *= config['anomalies']['contextual']['intensity'] * 0.3 
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'contextual'
                df.at[idx, 'confidence'] = 0.8
                total_anomalies += 1
        
        print(f"total contexual anomalies added at peak hours is : {total_anomalies}")
        if len(off_indices) > 0:
            selected_off = np.random.choice(off_indices, size=min(num_off_anomalies, len(off_indices)), replace=False)
            for idx in selected_off:
                # Make off hour values unexpectedly high
                df.at[idx, 'value'] *= config['anomalies']['contextual']['intensity'] * 2.5
                df.at[idx, 'is_anomaly'] = True
                df.at[idx, 'anomaly_type'] = 'contextual'
                df.at[idx, 'confidence'] = 0.8
                total_anomalies += 1
    
        print(f"total contexual anomalies added after off hours is : {total_anomalies}")

    # PATTERN ANOMALIES: Values that break sequential/temporal patterns
    # Example: Gradual drift, sudden changes in trend, sequence breaks
    if config['anomalies']['pattern']['enabled']:
        rate = config['anomalies']['pattern']['rate']
        n_pattern = int(len(df) * rate / 100)
        print(f" DEBUG INFO : pattern anomalies rate : {rate} , Number is : {n_pattern} out of {len(df)}")
        
        # Create pattern-breaking sequences
        for _ in range(n_pattern):
            start_idx = np.random.randint(10, len(df) - 10)
            pattern_length = np.random.randint(3, 8)  # Anomaly sequence length

            # New: Add a unique identifier for this pattern anomaly event
            anomaly_group_id = f"pattern_{total_anomaly_events}"
            
            for j in range(pattern_length):
                if start_idx + j < len(df):
                    if np.random.random() < 0.5:
                        # Gradual drift: slowly increasing deviation
                        drift_factor = config['anomalies']['pattern']['intensity'] * (j / pattern_length)
                        df.at[start_idx + j, 'value'] *= (1 + drift_factor)
                    else:
                        # Sudden change: immediate jump
                        df.at[start_idx + j, 'value'] *= config['anomalies']['pattern']['intensity']
                    
                    df.at[start_idx + j, 'is_anomaly'] = True
                    df.at[start_idx + j, 'anomaly_type'] = 'pattern'
                    df.at[start_idx + j, 'confidence'] = 0.7
                    df.at[start_idx + j, 'anomaly_group_id'] = anomaly_group_id # Assign the group ID

            # Important: Increment the event counter only once per pattern
            total_anomaly_events += 1
            total_anomalies += 1
    
    print(f"total contexual anomalies , pattern added after pattern anamoalies is : {total_anomalies}")

    # POINT ANOMALIES: Statistical outliers - individual points that are far from normal
    # Example: Sudden spikes, unexpected drops, sensor errors
    if config['anomalies']['point']['enabled']:
        rate = config['anomalies']['point']['rate']  # Percentage of individual points to make anomalous
        intensity = config['anomalies']['point']['intensity']  # How far from normal (multiplier)
        
        n_point = int(len(df) * rate / 100)
        point_indices = np.random.choice(df.index, size=n_point, replace=False)
        print(f" DEBUG INFO : point anomalies rate : {rate} , Number is : {n_point} out of {len(df)}")

        for idx in point_indices:
            # Random spike or drop
            if np.random.random() < 0.5:
                df.at[idx, 'value'] *= intensity  # Spike up
            else:
                df.at[idx, 'value'] *= (1 / intensity)  # Drop down
            
            df.at[idx, 'is_anomaly'] = True
            df.at[idx, 'anomaly_type'] = 'point'
            df.at[idx, 'confidence'] = 0.9  # Point anomalies are usually obvious
            total_anomalies += 1
    
    print(f"total added after point anamoalies is : {total_anomalies}")
    st.success(f"✅ Added {total_anomalies:,} synthetic anomalies to the dataset")
    return df

def convert_numpy_types(obj):
    """
    Recursively converts numpy data types in a dictionary or list to
    native Python data types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.floating)):
        return float(obj)
    else:
        return obj
