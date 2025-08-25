
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
from rule_based_methods import RuleBasedAnomaly
from ml_based_models import MLBasedAnomaly

def calculate_metrics(true_labels, predicted_labels):
    """Calculate comprehensive evaluation metrics"""
    try:

        if len(true_labels) == 0 or len(predicted_labels) == 0:
            return { 'accuracy' : 0, 'precision':0, 'recall':0, 'f1_score':0, 'true_positives':0, 'false_positives':0, 'true_negatives':0, 'false_negatives':0 }

        # Convert to numpy arrays
        y_true = np.array(true_labels).astype(int)
        y_pred = np.array(predicted_labels).astype(int)
            
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle case where no anomalies are predicted
        if np.sum(y_pred) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return { 'accuracy' : 0, 'precision':0, 'recall':0, 'f1_score':0, 'true_positives':0, 'false_positives':0, 'true_negatives':0, 'false_negatives':0 }


def calculate_detailed_metrics(df, predictions, method_name):
    """Calculate granular metrics by anomaly type and time"""
    results = {
        'overall': {},
        'by_anomaly_type': {},
        'by_time_period': {}
    }
    
    # Overall metrics
    if 'is_anomaly' in df.columns:
        ground_truth = df['is_anomaly'].values
        results['overall'] = calculate_metrics(ground_truth, predictions)

    # By anomaly type
    if 'anomaly_type' in df.columns:
        for anom_type in df['anomaly_type'].unique():
            mask = df['anomaly_type'] == anom_type
            if mask.sum() > 0:
                type_gt = df.loc[mask, 'is_anomaly'].values
                type_pred = predictions[mask]
                results['by_anomaly_type'][anom_type] = calculate_metrics(type_gt, type_pred)

    # By time period
    if 'hour' in df.columns:
        time_periods = {
            'peak_hours': df['is_peak_hour'] if 'is_peak_hour' in df.columns else df['hour'].isin([8,9,17,18]),
            'off_hours': df['is_off_hour'] if 'is_off_hour' in df.columns else df['hour'].isin([0,1,2,3,4,22,23]),
            'weekend': df['is_weekend'] if 'is_weekend' in df.columns else df['timestamp'].dt.dayofweek >= 5
        }
        
        for period_name, period_mask in time_periods.items():
            if period_mask.sum() > 0 and 'is_anomaly' in df.columns:
                period_gt = df.loc[period_mask, 'is_anomaly'].values
                period_pred = predictions[period_mask]
                results['by_time_period'][period_name] = calculate_metrics(period_gt, period_pred)
    
    return results
