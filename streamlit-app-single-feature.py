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
from synthetic_anomaly_generation import convert_numpy_types, get_all_anomalous_timestamps, generate_synthetic_data

# Set page config
st.set_page_config(
    page_title="Anomlytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ComprehensiveAnomalyStudio:
    def __init__(self):
        # Initialize session state variables
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        if 'modified_data' not in st.session_state:
            st.session_state.modified_data = None
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = {}
        if 'rule_results' not in st.session_state:
            st.session_state.rule_results = {}
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None

    def parse_time_ranges(self, time_ranges_str):
        """Parse time ranges from string format like '8-10,17-19' to list of hours"""
        try:
            hours = []
            ranges = time_ranges_str.split(',')
            for range_str in ranges:
                if '-' in range_str:
                    start, end = map(int, range_str.strip().split('-'))
                    hours.extend(list(range(start, end + 1)))
                else:
                    hours.append(int(range_str.strip()))
            return sorted(list(set([h for h in hours if 0 <= h <= 23])))
        except Exception as e:
            st.error(f"Invalid time range format: {e}")
            return []

    def create_interactive_plot(self, df, title="Time Series Data"):
        """Create interactive plot with enhanced point-by-point editing capability"""
        fig = go.Figure()
        
        # Normal points
        normal_df = df[~df['is_anomaly']]
        if len(normal_df) > 0:
            fig.add_trace(go.Scatter(
                x=normal_df['timestamp'],
                y=normal_df['value'],
                mode='lines+markers',
                name='Normal Data',
                line=dict(color='blue', width=1),
                marker=dict(size=3, color='blue'),
                customdata=normal_df.index,  # Store index for point identification
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<br>Type: Normal<br>Index: %{customdata}<extra></extra>'
            ))
        
        # Anomaly points by type
        if 'is_anomaly' in df.columns:
            anomaly_df = df[df['is_anomaly']]
            if len(anomaly_df) > 0:
                anomaly_types = anomaly_df['anomaly_type'].unique()
                colors = {'contextual': 'red', 'pattern': 'orange', 'point': 'purple'}
                
                for anom_type in anomaly_types:
                    type_df = anomaly_df[anomaly_df['anomaly_type'] == anom_type]
                    fig.add_trace(go.Scatter(
                        x=type_df['timestamp'],
                        y=type_df['value'],
                        mode='markers',
                        name=f'{anom_type.title()} Anomaly',
                        marker=dict(size=8, color=colors.get(anom_type, 'red'), symbol='x'),
                        customdata=type_df.index,
                        hovertemplate=f'<b>%{{x}}</b><br>Value: %{{y:.2f}}<br>Type: {anom_type.title()}<br>Index: %{{customdata}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Timestamp',
            yaxis_title='Value',
            hovermode='closest',
            showlegend=True,
            height=600,
            clickmode='event+select'  # Enable click events
        )
        
        return fig
    
    # ANOMALY DETECTION 

    
    def show_glossary(self):
        """Display comprehensive glossary of terms"""
        st.markdown("""
        ## 📖 Anomlytics Glossary**
        
        ### **⚙️ Data Synthesis Parameters**
        
        **Base Value**: The starting point for all calculations. If base=100, this is the "normal" value around which all variations occur.
        
        **Noise Level**: Random variation added to make data realistic. If noise=5, values randomly vary by ±5 around the calculated value.
        
        **Data Interval**: Time between consecutive data points (15, 30, or 60 minutes).
        
        ### **📅 Temporal Components**
        
        **Daily Strength**: How much values change throughout the day. Higher values = bigger differences between day/night.
        
        **Weekly Strength**: How much values change throughout the week. Higher values = bigger differences between weekdays/weekends.
        
        **Monthly Strength**: How much values change throughout the month. Higher values = bigger differences between month start/end.
        
        ### **Trend Multipliers**
        
        **Peak Hour Multiplier**: How much to multiply base values during busy periods (e.g., 1.5 = 50% higher).
        
        **Off Hour Multiplier**: How much to multiply base values during quiet periods (e.g., 0.7 = 30% lower).
        
        **Weekend Multiplier**: How much to multiply base values on weekends (e.g., 0.8 = 20% lower).
        
        **Weekday Specific**: Individual multipliers for each day of the week to capture weekly patterns.
        
        ### **🔎 Anomaly Taxonomy**
        
        **Contextual Anomalies**: Values that seem normal individually but are wrong for the time/context.
        - Example: Low website traffic during lunch hour, high traffic at 3 AM
        
        **Pattern Anomalies**: Values that break sequential/temporal patterns.
        - Example: Gradual increase when there should be decrease, sudden trend changes
        
        **Point Anomalies**: Individual values that are statistical outliers.
        - Example: Sudden spikes, sensor errors, unexpected drops
        
        ### **Anomaly Parameters**
        
        **Anomaly Rate (%)**: What percentage of data points should be anomalous (e.g., 2% = 2 out of every 100 points).
        
        **Intensity**: How far from normal the anomalies should be (e.g., 2.5 = 250% of normal value).
        
        ### **Detection Methods**
        
        **Rule-Based**: Use simple statistical rules (Z-score, IQR, moving averages).
        
        **ML-Based**: Use machine learning algorithms (Isolation Forest, One-Class SVM, DBSCAN).
        
        ### **Evaluation Metrics**
        
        **Precision**: Of all points flagged as anomalies, how many were actually anomalies?
        
        **Recall**: Of all actual anomalies, how many did we successfully detect?
        
        **F1-Score**: Harmonic mean of precision and recall (balanced measure).
        
        **Accuracy**: Overall percentage of correct classifications.
        """)

    def get_model_config_ui(self, model_name):
        """Generate UI configuration for each ML model"""
        config = {}
    
        if model_name == 'prophet':
            config['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'], key=f"{model_name}_seasonality", help=self.get_params_helptext(model_name, 'seasonality_mode'))
            config['interval_width'] = st.slider("Confidence Interval", 0.5, 0.99, 0.8, key=f"{model_name}_interval", help=self.get_params_helptext(model_name,'interval_width'))
            config['threshold_method'] = st.selectbox("Threshold Method", ['percentile', 'std', 'iqr'], key=f"{model_name}_threshold", help=self.get_params_helptext(model_name,'threshold_method'))
            config['threshold_value'] = st.slider("Threshold Value", 0.01, 0.2, 0.05, key=f"{model_name}_threshold_val", help=self.get_params_helptext(model_name,'Threshold Value'))
    
        elif model_name == 'arima':
            config['p'] = st.slider("AR Order (p)", 0, 5, 1, key=f"{model_name}_p")
            config['d'] = st.slider("Differencing (d)", 0, 2, 1, key=f"{model_name}_d")
            config['q'] = st.slider("MA Order (q)", 0, 5, 1, key=f"{model_name}_q")
            config['threshold_method'] = st.selectbox("Threshold Method", ['std', 'percentile'], key=f"{model_name}_threshold")
            config['threshold_value'] = st.slider("Threshold Multiplier", 1.0, 5.0, 2.0, key=f"{model_name}_threshold_val")
    
        elif model_name == 'sarima':
            config['p'] = st.slider("AR Order (p)", 0, 3, 1, key=f"{model_name}_p")
            config['d'] = st.slider("Differencing (d)", 0, 2, 1, key=f"{model_name}_d")
            config['q'] = st.slider("MA Order (q)", 0, 3, 1, key=f"{model_name}_q")
            config['P'] = st.slider("Seasonal AR (P)", 0, 2, 1, key=f"{model_name}_P")
            config['D'] = st.slider("Seasonal Diff (D)", 0, 1, 1, key=f"{model_name}_D")
            config['Q'] = st.slider("Seasonal MA (Q)", 0, 2, 1, key=f"{model_name}_Q")
            config['s'] = st.slider("Seasonal Period", 4, 24, 12, key=f"{model_name}_s")
            config['threshold_method'] = st.selectbox("Threshold Method", ['std', 'percentile'], key=f"{model_name}_threshold")
            config['threshold_value'] = st.slider("Threshold Multiplier", 1.0, 5.0, 2.0, key=f"{model_name}_threshold_val")
    
        elif model_name == 'auto_arima':
            config['seasonal'] = st.checkbox("Enable Seasonal", True, key=f"{model_name}_seasonal")
            config['stepwise'] = st.checkbox("Stepwise Search", True, key=f"{model_name}_stepwise")
            config['suppress_warnings'] = st.checkbox("Suppress Warnings", True, key=f"{model_name}_suppress")
            config['max_p'] = st.slider("Max AR Order", 3, 8, 5, key=f"{model_name}_max_p")
            config['max_q'] = st.slider("Max MA Order", 3, 8, 5, key=f"{model_name}_max_q")
            config['threshold_method'] = st.selectbox("Threshold Method", ['std', 'percentile'], key=f"{model_name}_threshold")
            config['threshold_value'] = st.slider("Threshold Multiplier", 1.0, 5.0, 2.0, key=f"{model_name}_threshold_val")

        elif model_name == 'isolation_forest':
            config['n_estimators'] = st.slider("Number of Trees", 50, 500, 100, key=f"{model_name}_estimators", help=self.get_params_helptext(model_name,'n_estimators'))
            config['contamination'] = st.slider("Contamination", 0.01, 0.5, 0.1, key=f"{model_name}_contamination", help=self.get_params_helptext(model_name,'contamination'))
            config['max_samples'] = st.selectbox("Max Samples", ['auto', 0.5, 0.7, 1.0], key=f"{model_name}_max_samples", help=self.get_params_helptext(model_name,'max_samples'))
            config['random_state'] = st.number_input("Random State", 0, 1000, 42, key=f"{model_name}_random_state", help=self.get_params_helptext(model_name,'random_state'))
    
        elif model_name == 'one_class_svm':
            config['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"{model_name}_kernel", help=self.get_params_helptext(model_name,'kernel'))
            config['gamma'] = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], key=f"{model_name}_gamma", help=self.get_params_helptext(model_name,'gamma'))
            config['nu'] = st.slider("Nu (Anomaly Fraction)", 0.01, 0.5, 0.05, key=f"{model_name}_nu", help=self.get_params_helptext(model_name,'nu'))
            config['degree'] = st.slider("Polynomial Degree", 2, 5, 3, key=f"{model_name}_degree", help=self.get_params_helptext(model_name,'degree'))
    
        elif model_name == 'elliptic_envelope':
            config['contamination'] = st.slider("Contamination", 0.01, 0.5, 0.1, key=f"{model_name}_contamination",  help=self.get_params_helptext(model_name,'contamination'))
            config['support_fraction'] = st.slider("Support Fraction", 0.5, 1.0, 0.8, key=f"{model_name}_support_fraction", help=self.get_params_helptext(model_name,'support_fraction'))
            config['random_state'] = st.number_input("Random State", 0, 1000, 42, key=f"{model_name}_random_state", help=self.get_params_helptext(model_name,'random_state'))
    
        elif model_name == 'local_outlier_factor':
            config['n_neighbors'] = st.slider("Number of Neighbors", 5, 50, 20, key=f"{model_name}_neighbors", help=self.get_params_helptext(model_name,'n_neighbors'))
            config['contamination'] = st.slider("Contamination", 0.01, 0.5, 0.1, key=f"{model_name}_contamination", help=self.get_params_helptext(model_name,'contamination'))
            config['algorithm'] = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], key=f"{model_name}_algorithm", help=self.get_params_helptext(model_name,'algorithm'))
            config['leaf_size'] = st.slider("Leaf Size", 10, 50, 30, key=f"{model_name}_leaf_size", help=self.get_params_helptext(model_name,'leaf_size'))
    
        elif model_name == 'dbscan':
            config['eps'] = st.slider("Epsilon (Neighborhood)", 0.1, 2.0, 0.5, key=f"{model_name}_eps", help=self.get_params_helptext(model_name,'eps'))
            config['min_samples'] = st.slider("Min Samples", 2, 20, 5, key=f"{model_name}_min_samples", help=self.get_params_helptext(model_name,'min_samples'))
            config['metric'] = st.selectbox("Distance Metric", ['euclidean', 'manhattan', 'cosine'], key=f"{model_name}_metric", help=self.get_params_helptext(model_name,'metric'))
            config['algorithm'] = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], key=f"{model_name}_algorithm", help=self.get_params_helptext(model_name,'algorithm'))
    
        elif model_name == 'kmeans':
            config['n_clusters'] = st.slider("Number of Clusters", 2, 20, 8, key=f"{model_name}_clusters", help=self.get_params_helptext(model_name,'n_clusters'))
            config['init'] = st.selectbox("Initialization", ['k-means++', 'random'], key=f"{model_name}_init", help=self.get_params_helptext(model_name,'init'))
            config['n_init'] = st.slider("Number of Initializations", 1, 20, 10, key=f"{model_name}_n_init", help=self.get_params_helptext(model_name,'n_init'))
            config['max_iter'] = st.slider("Max Iterations", 100, 1000, 300, key=f"{model_name}_max_iter", help=self.get_params_helptext(model_name,'max_iter'))
            config['contamination'] = st.slider("Anomaly Threshold", 0.01, 0.5, 0.1, key=f"{model_name}_contamination", help=self.get_params_helptext(model_name,'contamination'))
            config['random_state'] = st.number_input("Random State", 0, 1000, 42, key=f"{model_name}_random_state", help=self.get_params_helptext(model_name,'random_state'))
    
        elif model_name == 'zscore':
            config['threshold'] = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key=f"{model_name}_threshold", help=self.get_params_helptext(model_name,'threshold'))
            config['window_size'] = st.slider("Rolling Window Size", 5, 100, 30, key=f"{model_name}_window", help=self.get_params_helptext(model_name,'window_size'))
            config['use_modified'] = st.checkbox("Use Modified Z-Score", False, key=f"{model_name}_modified", help=self.get_params_helptext(model_name,'use_modified'))
    
        elif model_name == 'mad':
            config['threshold'] = st.slider("MAD Threshold", 1.0, 5.0, 3.5, key=f"{model_name}_threshold",  help=self.get_params_helptext(model_name,'threshold'))
            config['window_size'] = st.slider("Rolling Window Size", 5, 100, 30, key=f"{model_name}_window", help=self.get_params_helptext(model_name,'window_size'))
            config['constant'] = st.slider("Consistency Constant", 1.0, 2.0, 1.4826, key=f"{model_name}_constant", help=self.get_params_helptext(model_name,'constant'))
    
        elif model_name == 'ewma':
            config['alpha'] = st.slider("Smoothing Factor (Alpha)", 0.01, 1.0, 0.3, key=f"{model_name}_alpha", help=self.get_params_helptext(model_name,'alpha'))
            config['threshold'] = st.slider("Threshold", 0.1, 8.0, 3.0, key=f"{model_name}_threshold", help=self.get_params_helptext(model_name,'threshold'))
            config['lambda_param'] = st.slider("EWMA Lambda", 0.05, 2.0, 0.1, key=f"{model_name}_lambda", help=self.get_params_helptext(model_name,'lambda_param'))
            config['startup_periods'] = st.slider("Startup Periods", 5, 50, 20, key=f"{model_name}_startup", help=self.get_params_helptext(model_name,'startup_periods'))
    
        elif model_name == 'exponential_smoothing':
            config['trend'] = st.selectbox("Trend Component", ['add', 'mul', None], key=f"{model_name}_trend", help=self.get_params_helptext(model_name,'trend'))
            config['seasonal'] = st.selectbox("Seasonal Component", ['add', 'mul', None], key=f"{model_name}_seasonal", help=self.get_params_helptext(model_name,'seasonal'))
            config['seasonal_periods'] = st.slider("Seasonal Periods", 4, 24, 12, key=f"{model_name}_seasonal_periods", help=self.get_params_helptext(model_name,'seasonal_periods'))
            config['threshold_method'] = st.selectbox("Threshold Method", ['std', 'percentile'], key=f"{model_name}_threshold", help=self.get_params_helptext(model_name,'threshold_method'))
            config['threshold_value'] = st.slider("Threshold Multiplier", 1.0, 5.0, 2.0, key=f"{model_name}_threshold_val", help=self.get_params_helptext(model_name,'threshold_value'))
    
        elif model_name == 'lstm':
            config['sequence_length'] = st.slider("Sequence Length", 10, 100, 50, key=f"{model_name}_seq_len")
            config['hidden_size'] = st.slider("Hidden Units", 32, 256, 64, key=f"{model_name}_hidden")
            config['num_layers'] = st.slider("LSTM Layers", 1, 5, 2, key=f"{model_name}_layers")
            config['epochs'] = st.slider("Training Epochs", 10, 200, 50, key=f"{model_name}_epochs")
            config['batch_size'] = st.slider("Batch Size", 16, 128, 32, key=f"{model_name}_batch")
            config['learning_rate'] = st.selectbox( "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], format_func=lambda x: f"{x} ({'Conservative' if x==0.0001 else 'Slow' if x==0.0005 else 'Recommended' if x==0.001 else 'Fast' if x==0.005 else 'Aggressive'})", index=2, key=f"{model_name}_lr")
            config['threshold_method'] = st.selectbox("Threshold Method", ['percentile', 'std'], key=f"{model_name}_threshold")
            if config['threshold_method'] == "percentile":
                config['threshold_value'] = st.slider("Threshold Value", 0.01, 0.2, 0.05, help="0.05 = top 5% are anomalies" , key=f"{model_name}_threshold_val")
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
            else:
                config['threshold_value'] = st.slider("Standard Deviation Multiplier", 0.01, 0.20, 0.05)
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
    
        elif model_name == 'gru':
            config['sequence_length'] = st.slider("Sequence Length", 10, 100, 50, key=f"{model_name}_seq_len")
            config['hidden_size'] = st.slider("Hidden Units", 32, 256, 64, key=f"{model_name}_hidden")
            config['num_layers'] = st.slider("GRU Layers", 1, 5, 2, key=f"{model_name}_layers")
            config['epochs'] = st.slider("Training Epochs", 10, 200, 50, key=f"{model_name}_epochs")
            config['batch_size'] = st.slider("Batch Size", 16, 128, 32, key=f"{model_name}_batch")
            config['learning_rate'] = st.selectbox( "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], format_func=lambda x: f"{x} ({'Conservative' if x==0.0001 else 'Slow' if x==0.0005 else 'Recommended' if x==0.001 else 'Fast' if x==0.005 else 'Aggressive'})", index=2, key=f"{model_name}_lr")
            config['threshold_method'] = st.selectbox("Threshold Method", ['percentile', 'std'], key=f"{model_name}_threshold")
            if config['threshold_method'] == "percentile":
                config['threshold_value'] = st.slider("Threshold Value", 0.01, 0.2, 0.05, help="0.05 = top 5% are anomalies" , key=f"{model_name}_threshold_val")
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
            else:
                config['threshold_value'] = st.slider("Standard Deviation Multiplier", 0.01, 0.20, 0.05)
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
    
        elif model_name == 'usad':
            config['sequence_length'] = st.slider("Sequence Length", 10, 100, 50, key=f"{model_name}_seq_len")
            config['hidden_size'] = st.slider("Hidden Units", 32, 256, 100, key=f"{model_name}_hidden")
            config['epochs'] = st.slider("Training Epochs", 50, 300, 100, key=f"{model_name}_epochs")
            config['batch_size'] = st.slider("Batch Size", 16, 128, 64, key=f"{model_name}_batch")
            config['learning_rate'] = st.selectbox( "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], format_func=lambda x: f"{x} ({'Conservative' if x==0.0001 else 'Slow' if x==0.0005 else 'Recommended' if x==0.001 else 'Fast' if x==0.005 else 'Aggressive'})", index=2, key=f"{model_name}_lr")
            config['alpha'] = st.slider("Alpha (Loss Weight)", 0.1, 1.0, 0.5, key=f"{model_name}_alpha")
            config['beta'] = st.slider("Beta (Loss Weight)", 0.1, 1.0, 0.5, key=f"{model_name}_beta")
            config['threshold_method'] = st.selectbox("Threshold Method", ['percentile', 'std'], key=f"{model_name}_threshold")
            if config['threshold_method'] == "percentile":
                config['threshold_value'] = st.slider("Threshold Value", 0.01, 0.2, 0.05, help="0.05 = top 5% are anomalies" , key=f"{model_name}_threshold_val")
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
            else:
                config['threshold_value'] = st.slider("Standard Deviation Multiplier", 0.01, 0.20, 0.05)
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
    
        elif model_name == 'autoencoder':
            config['encoding_dim'] = st.slider("Encoding Dimension", 8, 128, 32, key=f"{model_name}_encoding_dim")
            #config['hidden_layers'] = st.multiselect("Hidden Layer Sizes", [16, 32, 64, 128, 256], default=[64, 32], key=f"{model_name}_hidden")
            config['hidden_layers'] = st.multiselect("Hidden Layer Sizes", [16, 32, 50, 100, 64, 128, 256], default=[50, 100], key=f"{model_name}_hidden")
            #config['hidden_layers'] = st.selectbox("Hidden Layer Configuration", ["32,16", "64,32", "100,50", "128,64"], default=[64, 32], key=f"{model_name}_hidden")
            config['activation'] = st.selectbox("Activation Function", ['relu', 'tanh', 'sigmoid'], key=f"{model_name}_activation")
            config['epochs'] = st.slider("Training Epochs", 50, 500, 100, key=f"{model_name}_epochs")
            config['batch_size'] = st.slider("Batch Size",  16, 128, 32, key=f"{model_name}_batch")
            config['learning_rate'] = st.selectbox( "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], format_func=lambda x: f"{x} ({'Conservative' if x==0.0001 else 'Slow' if x==0.0005 else 'Recommended' if x==0.001 else 'Fast' if x==0.005 else 'Aggressive'})", index=2, key=f"{model_name}_lr")
            config['validation_split'] = st.slider("Validation Split", 0.1, 0.3, 0.2, key=f"{model_name}_val_split")
            config['threshold_method'] = st.selectbox("Threshold Method", ['percentile', 'std'], key=f"{model_name}_threshold")
            if config['threshold_method'] == "percentile":
                config['threshold_value'] = st.slider("Threshold Value", 0.01, 0.2, 0.05, help="0.05 = top 5% are anomalies" , key=f"{model_name}_threshold_val")
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
            else:
                config['threshold_value'] = st.slider("Standard Deviation Multiplier", 0.01, 0.20, 0.05)
                config['percentile_threshold'] = (1 - config['threshold_value']) * 100  
    
        else:
            # Default config for unknown models
            config['threshold'] = st.slider("Threshold", 0.01, 1.0, 0.1, key=f"{model_name}_threshold")
    
        return config

    def get_params_helptext(self, model_name, param_name):
        helptext = ""
        if model_name == 'isolation_forest':
            if param_name == 'n_estimators':
                helptext = "The number of isolation trees to build. More trees can improve accuracy but increase computation time."
            if param_name == 'max_features':
                helptext = "The number of features to consider when looking for the best split. A smaller number adds more randomness."
            if param_name == 'contamination':
                helptext = "The expected proportion of anomalies in the dataset. This parameter is crucial for setting the threshold for the anomaly score."
            if param_name == 'random_state':
                helptext = "Controls the randomness of the model for reproducible results. Setting it to an integer ensures the same trees and splits are generated each time."
        if model_name == 'one_class_svm':
            if param_name == 'kernel': 
                helptext = "The kernel function to use. The 'rbf' (Radial Basis Function) kernel is a common choice for non-linear data."
            if param_name == 'gamma':
                helptext = "Kernel coefficient for the 'rbf' kernel. A small gamma means a broader decision boundary, while a large gamma creates a tighter one."
            if param_name == 'nu': 
                helptext = "An upper bound on the fraction of training errors and a lower bound on the fraction of support vectors. A value between 0 and 1."
            if param_name == 'degree': 
                helptext = "The degree of the polynomial kernel function ('poly'). It is ignored by all other kernels and must be non-negative."
        if model_name == 'elliptic_envelope':
            if param_name == 'contamination':
                helptext = "The proportion of outliers in the dataset. This parameter directly sets the threshold for classifying anomalies."
            if param_name == 'support_fraction':
                helptext = "The proportion of points to be included in the support of the raw estimate. This parameter is used to find a robust estimate of the covariance."
            if param_name == 'random_state':
                helptext = "Determines the pseudo-random number generator for shuffling the data, ensuring reproducible results."
        if model_name == 'local_outlier_factor':
            if param_name == 'n_neighbors':
                helptext = "The number of neighbors to consider for each sample. This defines the local density around a data point."
            if param_name == 'contamination':
                helptext = "The proportion of anomalies in the dataset. This parameter is used to set the final threshold."
            if param_name == 'algorithm':
                helptext = "The algorithm used to compute the nearest neighbors. Options include 'ball_tree', 'kd_tree', 'brute', and 'auto'."
            if param_name == 'leaf_size':
                helptext = "The size of the leaf nodes in the tree-based search algorithms. This affects the speed and memory usage."
        if model_name == 'dbscan':
            if param_name == 'eps':
                helptext = "The maximum distance between two samples for one to be considered in the neighborhood of the other."
            if param_name == 'min_samples':
                helptext = "The number of samples in a neighborhood for a point to be considered as a core point."
            if param_name == 'metric':
                helptext = "The distance metric used to measure the distance between data points. 'Euclidean' is the default and most common."
            if param_name == 'algorithm':
                helptext = "The algorithm used to compute the nearest neighbors. Common choices include 'auto', 'kd_tree', and 'ball_tree'."
        if model_name == 'kmeans':
            if param_name == 'n_clusters':
                helptext = "The number of clusters to form. Anomalies are often far from any cluster centroid."
            if param_name == 'init':
                helptext = "The method for initializing the cluster centroids. 'k-means++' is a smart initialization that speeds up convergence. 'random' is a simple random choice."
            if param_name == 'n_init':
                helptext = "The number of times the K-Means algorithm will be run with different centroid seeds. The best result is kept."
            if param_name == 'max_iter':
                helptext = "The maximum number of iterations for a single run of the K-Means algorithm."
            if param_name == 'contamination':
                helptext = "The expected proportion of anomalies in the dataset. Used to set the decision threshold for the anomaly score."
            if param_name == 'random_state':
                helptext = "Determines the random number generation for centroid initialization, ensuring reproducibility."
        if model_name == 'zscore':
            if param_name == 'threshold':
                helptext = "The number of standard deviations a data point must be from the mean to be considered an anomaly."
            if param_name == 'window_size':
                helptext = "The number of previous data points to use to calculate the rolling mean and standard deviation."
            if param_name == 'use_modified':
                helptext = "A boolean flag. When set to 'True', it uses the modified Z-score based on the median and MAD (Median Absolute Deviation), which is more robust to outliers."
        if model_name == 'ewma':
            if param_name == 'alpha':
                helptext = "The smoothing parameter that controls the weight given to the most recent data point. A value closer to 1 makes the EWMA more responsive to recent changes."
            if param_name == 'threshold':
                helptext = "The maximum acceptable deviation from the EWMA value."
            if param_name == 'lambda_param':
                helptext = "The decay parameter used to calculate weights. This parameter is an alternative to alpha"
            if param_name == 'startup_periods':
                helptext = "The number of initial periods to use for the first calculation of the EWMA."
        if model_name == 'prophet':
            if param_name == 'seasonality_mode':
                helptext = "The type of seasonality. 'additive' is common for constant seasonality, while 'multiplicative' is for seasonality that changes with trend."
            if param_name == 'interval_width':
                helptext = "The width of the uncertainty interval. For anomaly detection, this interval defines the normal range, and anything outside is an anomaly."
            if param_name == 'threshold_method':
                helptext = "The method used to set the anomaly threshold. It can be a static value or based on the interval width."
            if param_name == 'threshold_value':
                helptext = "The numerical value used as a static threshold for anomaly detection."
            if param_name == 'changepoint_prior_scale':
                helptext = "The flexibility of the trend. A larger value allows the model to find more trend changes."
        if model_name == 'lstm' or model_name == 'gru':
            if param_name == 'sequence_length':
                helptext = ''
            if param_name == 'hidden_size':
                helptext = "The number of hidden units (neurons) in each LSTM / GRU layer. Increasing this can help the model learn more complex patterns."
            if param_name == 'num_layers':
                helptext = "The number of LSTM / GRU  layers stacked on top of each other. A deeper network can learn more abstract representations."
            if param_name == 'epochs':
                helptext = ''
            if param_name == 'batch_size':
                helptext = ''
            if param_name == 'learning_rate':
                helptext = ''
            if param_name == 'threshold_method':
                helptext = "The method for determining the anomaly threshold. 'percentile' flags a certain percentage of highest errors, while 'std' uses a standard deviation multiplier."
            if param_name == 'threshold_value':
                helptext = "If threshold_method is 'percentile', this represents the top percentage of errors considered anomalies (e.g., 0.05 = top 5%). For 'std', this is the multiplier for the standard deviation"
            if param_name == 'percentile_threshold':
                helptext = "The calculated percentile threshold based on the threshold_value (e.g., a 0.05 value corresponds to the 95th percentile)."
        if model_name == 'autoencoder':
            if param_name == 'encoding_dim':
                helptext = "The size of the hidden bottleneck layer. This forces the model to learn a compressed representation of the data. A smaller value can improve anomaly detection by preventing the model from memorizing anomalous patterns."
            if param_name == 'hidden_layers':
                helptext = "The number of neurons in the hidden layers of the encoder and decoder. You can specify multiple layers to create a deeper network."
            if param_name == 'activation':
                helptext = "The activation function to use in the hidden layers. Common choices include 'relu' (Rectified Linear Unit), 'tanh' (hyperbolic tangent), and 'sigmoid'."
            if param_name == 'epochs':
                helptext = ''
            if param_name == 'batch_size':
                helptext = ''
            if param_name == 'learning_rate':
                helptext = ''
            if param_name == 'validation_split':
                helptext = "The proportion of the training data to use for validation during training. This helps monitor for overfitting."
            if param_name == 'threshold_method':
                helptext = "The method for determining the anomaly threshold. 'percentile' flags a certain percentage of highest errors, while 'std' uses a standard deviation multiplier."
            if param_name == 'threshold_value':
                helptext = "If threshold_method is 'percentile', this represents the top percentage of errors considered anomalies (e.g., 0.05 = top 5%). For 'std', this is the multiplier for the standard deviation"
            if param_name == 'percentile_threshold':
                helptext = "The calculated percentile threshold based on the threshold_value (e.g., a 0.05 value corresponds to the 95th percentile)."

        if model_name == 'usad':
            if param_name == 'sequence_length':
                helptext = ''
            if param_name == 'hidden_size':
                helptext = "The number of hidden units (neurons) in each LSTM layer. Increasing this can help the model learn more complex patterns."
            if param_name == 'epochs':
                helptext = "The number of training epochs. More epochs can improve performance but increase the risk of overfitting."
            if param_name == 'batch_size':
                helptext = ''
            if param_name == 'learning_rate':
                helptext = ''
            if param_name == 'alpha':
                helptext = "A weighting parameter for the reconstruction loss of the first autoencoder. A higher value prioritizes accurate reconstruction."
            if param_name == 'beta':
                helptext = f"A weighting parameter for the adversarial loss of the two autoencoders. A higher value prioritizes the model's ability to discriminate between original and reconstructed data."
            if param_name == 'threshold_method':
                helptext = "The method for determining the anomaly threshold. 'percentile' flags a certain percentage of highest errors, while 'std' uses a standard deviation multiplier."
            if param_name == 'threshold_value':
                helptext = "If threshold_method is 'percentile', this represents the top percentage of errors considered anomalies (e.g., 0.05 = top 5%). For 'std', this is the multiplier for the standard deviation"


        return helptext
    
    
    

    def get_rule_helptext(self, rule_name):
        """Generate Help Text to be displayed for the rule"""
        helptext = ""
        if rule_name == 'sharp_jump_rule':
            helptext = "Detects sharp jumps/downs between the successive data-samples"
        if rule_name == 'sharp_jump_rule':
            helptext = "Detects unexpected trend"
        if rule_name == 'baseline_deviation_rule':
            helptext = "Detects standard deviation from the baseline"
        if rule_name == 'zscore_threshold':
            helptext = "Uses median-based Z-score"
        if rule_name == 'iqr_method':
            helptext = "Uses IQR-based outlier detection"
        if rule_name == 'moving_avg':
            helptext = "Uses the moving average deviation "
        if rule_name == 'static_min_threshold':
            helptext = "Values below the min_threshold are anomalies "
        if rule_name == 'static_max_threshold':
            helptext = "Values above the max_threshold are anomalies "
        if rule_name == 'static_range_rule':
            helptext = "Values beyond the range are anomalies "
        if rule_name == 'static_percentage_rule':
            helptext = "Value's percentage beyond the threshold_percentage is anomaly "
        if rule_name == 'rate_of_change':
            helptext = "rate of change of the window, is beyond the threshold "
        if rule_name == 'consecutive_anomaly':
            helptext = "Number of consecutive anomalies to be actually flagged as Anomaly"
        if rule_name == 'seasonal_decomp_rule':
            helptext = "deviations from seasonal pattern flagged as Anomaly"
        if rule_name == 'periodic_rule':
            helptext = "Detects shift_detection:timing shifts in periodic patterns, missing_periodicity: when periodic patterns disappear, frequency_domain: changes in frequency characteristics from seasonal pattern flagged as Anomaly"
        if rule_name == 'percentile_rule':
            helptext = "Detects deviations beyond the expected percentile"
        return helptext

    def get_rule_config_ui(self, rule_name):
        """Generate UI configuration for each rule-based method"""
        config = {}
        defaults = RuleBasedAnomaly.RULE_DEFAULTS.get(rule_name, {})
    
        if rule_name == 'sharp_jump_rule':
            config['jump_threshold'] = st.slider( "Jump Threshold Multiplier", defaults['min_jump_threshold'], defaults['max_jump_threshold'], defaults['jump_threshold'], defaults['step'], key=f"{rule_name}_jump_threshold", help="Multiplier for detecting sharp jumps (higher = less sensitive)")
            config['direction'] = st.selectbox( "Jump Direction", ['both', 'up', 'down'], key=f"{rule_name}_direction", help="Detect jumps in which direction")
    
        elif rule_name == 'trend_rule':
            config['trend_window'] = st.slider( "Trend Window Size", defaults['min_trend_window'], defaults['max_trend_window'], defaults['trend_window'], defaults['step'], key=f"{rule_name}_window", help="Number of points to analyze for trend detection")
            config['trend_threshold'] = st.slider( "Trend Correlation Threshold", 0.1, 1.0, defaults['trend_threshold'], 0.05, key=f"{rule_name}_correlation", help="Minimum correlation to consider as a trend")
            config['trend_type'] = st.selectbox( "Trend Type", ['increasing', 'decreasing', 'both'], key=f"{rule_name}_type")
    
        elif rule_name == 'baseline_deviation_rule': 
            config['baseline_window'] = st.slider( "Baseline Window Size", defaults['min_baseline_window'], defaults['max_baseline_window'], defaults['baseline_window'], defaults['step'], key=f"{rule_name}_window", help="Number of points for baseline calculation")
            config['deviation_threshold'] = st.slider( "Deviation Threshold", 1.0, 5.0, defaults['deviation_threshold'], 0.1, key=f"{rule_name}_deviation", help="Standard deviations from baseline") 
            config['baseline_method'] = st.selectbox( "Baseline Method", ['mean', 'median', 'mode'], key=f"{rule_name}_method")
    
        elif rule_name == 'zscore_threshold': 
            config['zscore_value'] = st.slider( "Z-Score Threshold", defaults['min_zscore'], defaults['max_zscore'], defaults['zscore_value'], defaults['step'], key=f"{rule_name}_threshold", help="Standard deviations from mean")
            config['use_modified'] = st.checkbox( "Use Modified Z-Score", defaults['use_modified'], key=f"{rule_name}_modified", help="Use median-based modified Z-score")
    
        elif rule_name == 'iqr_method': 
            config['iqr_multiplier'] = st.slider( "IQR Multiplier", defaults['min_iqr_multiplier'], defaults['max_iqr_multiplier'], defaults['iqr_multiplier'], defaults['step'], key=f"{rule_name}_multiplier", help="Multiplier for IQR-based outlier detection")
            config['use_median'] = st.checkbox( "Use Median for Center", defaults['use_median'], key=f"{rule_name}_median", help="Use median instead of quartiles for center")
      
        elif rule_name == 'moving_avg':
            config['moving_window'] = st.slider( "Moving Average Window", defaults['min_moving_window'], defaults['max_moving_window'], defaults['moving_window'], defaults['step'], key=f"{rule_name}_window", help="Window size for moving average")
            config['moving_threshold'] = st.slider( "Deviation Threshold (%)", defaults['min_threshold_pct'], defaults['max_threshold_pct'], defaults['moving_threshold'], defaults['threshold_step'], key=f"{rule_name}_threshold", help="Percentage deviation from moving average")
            config['center_window'] = st.checkbox( "Center Window", True, key=f"{rule_name}_center", help="Center the moving average window")
    
        elif rule_name == 'static_min_threshold':
            config['min_value'] = st.number_input( "Minimum Acceptable Value", value=defaults.get('min_value', 0.0), key=f"{rule_name}_min_val", help="Values BELOW this threshold are anomalies")
            config['inclusive'] = st.checkbox( "Include Boundary Value", True, key=f"{rule_name}_inclusive", help="Values equal to threshold are also anomalies")

    
        elif rule_name == 'static_max_threshold':
            config['max_value'] = st.number_input( "Maximum Acceptable Value", value=defaults.get('max_value', 100.0), key=f"{rule_name}_max_val", help="Values ABOVE this threshold are anomalies")
            config['inclusive'] = st.checkbox( "Include Boundary Value", True, key=f"{rule_name}_inclusive", help="Values equal to threshold are also anomalies")
            print(f"after config setting: static_max_threshold : max_value is {config['max_value']}")
    
        elif rule_name == 'static_range_rule':
            config['lower_bound'] = st.number_input( "Lower Bound", defaults['min_range'], defaults['max_range'], defaults['lower_bound'], defaults['step'], key=f"{rule_name}_lower", help="Lower bound of acceptable range")
            config['upper_bound'] = st.number_input( "Upper Bound", defaults['min_range'], defaults['max_range'], defaults['upper_bound'], defaults['step'], key=f"{rule_name}_upper", help="Upper bound of acceptable range")
            config['inclusive'] = st.checkbox( "Inclusive Bounds", True, key=f"{rule_name}_inclusive", help="Include boundary values as normal")
    
        elif rule_name == 'static_percentage_rule':
            config['min_value'] = st.number_input( "Minimum Value of Range", value=0.0, key=f"{rule_name}_min_val", help="Minimum value of the acceptable range (e.g., 0 for CPU %)")
            config['max_value'] = st.number_input( "Maximum Value of Range", value=100.0, key=f"{rule_name}_max_val", help="Maximum value of the acceptable range (e.g., 100 for CPU %, 200 for custom range)")
            config['threshold_percentage'] = st.slider("Threshold Percentage", 1.0, 100.0, 94.0, 1.0, key=f"{rule_name}_threshold_pct", help="Percentage of max range - values above this % are anomalies")
        
            # Show calculated threshold value for clarity
            min_val = config['min_value']
            max_val = config['max_value']
            threshold_pct = config['threshold_percentage']
            calculated_threshold = min_val + ((max_val - min_val) * threshold_pct / 100.0)
            st.info(f"📊 Anomaly threshold will be: **{calculated_threshold:.2f}** ({threshold_pct}% of range [{min_val}, {max_val}])")

    
        elif rule_name == 'rate_of_change':
            config['rate_threshold'] = st.slider( "Rate of Change Threshold", defaults['min_rate'], defaults['max_rate'], defaults['rate_threshold'], defaults['step'], key=f"{rule_name}_rate", help="Maximum allowed rate of change per unit time")
            config['window_size'] = st.slider( "Rate Calculation Window", 2, 10, defaults['window_size'], 1, key=f"{rule_name}_window", help="Number of points for rate calculation")
    
        elif rule_name == 'consecutive_anomaly':
            config['consecutive_count'] = st.slider( "Required Consecutive Count", 2, 10, defaults.get('consecutive_count', 3), 1, key=f"{rule_name}_count", help="Number of consecutive anomalous points required")
            config['base_method'] = st.selectbox( "Base Anomaly Detection Method", ['zscore', 'iqr', 'deviation'], key=f"{rule_name}_method", help="Method to detect individual anomalies before checking consecutiveness")
            config['base_threshold'] = st.slider( "Base Threshold", 1.0, 5.0, defaults.get('base_threshold', 2.0), 0.1, key=f"{rule_name}_base", help="Threshold for base anomaly detection method")
    
        elif rule_name == 'seasonal_decomp_rule':
            config['seasonal_period'] = st.slider( "Seasonal Period", defaults['min_period'], defaults['max_period'], defaults['seasonal_period'], defaults['step'], key=f"{rule_name}_period", help="Length of seasonal cycle (e.g., 24 for daily, 168 for weekly)")
            config['seasonal_threshold'] = st.slider( "Seasonal Threshold", 1.0, 5.0, defaults['seasonal_threshold'], 0.1, key=f"{rule_name}_threshold", help="Standard deviations from seasonal pattern")

        elif rule_name == 'periodic_rule':
            config['period'] = st.slider("Period", defaults['min_period'], defaults['max_period'], defaults['period'], defaults['step'],key=f"{rule_name}_period", help="Length of periodic cycle (e.g., 24 for daily, 168 for weekly)")
            config['threshold'] = st.slider( "Base Threshold", 1.0, 5.0, defaults['threshold'], 0.1, key=f"{rule_name}_threshold", help="Base threshold for anomaly detection")
            # Selection variable - dropdown for rule type
            periodic_rule_types = { 'shift_detection': 'Shift Detection - Detects timing shifts in periodic patterns', 'missing_periodicity': 'Missing Periodicity - Detects when periodic patterns disappear', 'frequency_domain': 'Frequency Domain - Detects changes in frequency characteristics' }
            config['periodic_rule_type'] = st.selectbox( "Periodic Rule Type", options=list(periodic_rule_types.keys()), format_func=lambda x: periodic_rule_types[x], index=list(periodic_rule_types.keys()).index(defaults.get('periodic_rule_type', 'shift_detection')), key=f"{rule_name}_type", help="Select the type of periodic anomaly detection to apply")
            # Dynamic variables - shown based on selected rule type
            selected_type = config['periodic_rule_type']
            if selected_type == 'shift_detection':
                config['shift_threshold'] = st.slider( "Shift Threshold", 0.1, 1.0, defaults.get('shift_threshold', 0.3), 0.05, key=f"{rule_name}_shift_threshold", help="Correlation threshold for detecting period shifts (lower = more sensitive)")
            elif selected_type == 'missing_periodicity':
                config['min_correlation'] = st.slider( "Minimum Correlation", 0.1, 1.0, defaults.get('min_correlation', 0.7), 0.05, key=f"{rule_name}_min_correlation", help="Minimum correlation to maintain periodicity (higher = stricter)")
                config['window_size'] = st.slider( "Analysis Window Size", config['period'] * 2, config['period'] * 8, defaults.get('window_size', config['period'] * 4), config['period'], key=f"{rule_name}_window_size", help="Size of window for periodicity analysis (in data points)")
            elif selected_type == 'frequency_domain':
                config['frequency_threshold'] = st.slider( "Frequency Threshold", 0.5, 5.0, defaults.get('frequency_threshold', 2.0), 0.1, key=f"{rule_name}_freq_threshold", help="Threshold for frequency domain changes (higher = less sensitive)")
                config['window_size'] = st.slider( "FFT Window Size", config['period'] * 2, config['period'] * 8, defaults.get('window_size', config['period'] * 4), config['period'], key=f"{rule_name}_fft_window_size", help="Window size for FFT analysis (should be multiple of period)")
                config['num_periods'] = st.slider( "Number of Periods", 2, 10, defaults.get('num_periods', 4), 1, key=f"{rule_name}_num_periods", help="Number of periods to analyze for baseline")

        elif rule_name == 'percentile_rule':
            config['lower_percentile'] = st.slider( "Lower Percentile", defaults['min_percentile'], 50.0, defaults['lower_percentile'], defaults['step'], key=f"{rule_name}_lower_pct", help="Lower percentile boundary")
            config['upper_percentile'] = st.slider( "Upper Percentile", 50.0, defaults['max_percentile'], defaults['upper_percentile'], defaults['step'], key=f"{rule_name}_upper_pct", help="Upper percentile boundary")
    
        else:
            # Default config for unknown rules
            config['threshold'] = st.slider( "Threshold", 0.01, 10.0, 1.0, 0.1, key=f"{rule_name}_threshold")
    
        return config

# Enhanced Data Generation Visualization with Injected Anomaly Details
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def extract_generated_anomalies(df):
    """
    Extract anomaly information from generated dataframe
    
    Args:
        df: Generated dataframe with columns:
            - 'is_anomaly': boolean indicating if point is anomaly
            - 'anomaly_type': type of anomaly ('contextual', 'pattern', 'point')
            - 'value': current value (potentially modified by anomaly injection)
            - 'original_value': original value before anomaly injection (optional)
            - 'pattern_group_id': group ID for pattern anomalies (for patterns)
            - 'anomaly_id': unique identifier for each anomaly instance
    
    Returns:
        dict: Generated anomalies in the format expected by the UI
    """
    generated_anomalies = {
        'contextual': {'count': 0, 'timestamps': [], 'original_values': []},
        'pattern': {'count': 0, 'timestamps': [], 'original_values': []},
        'point': {'count': 0, 'timestamps': [], 'original_values': []}
    }
    
    # Check if anomaly columns exist
    if 'is_anomaly' not in df.columns:
        return generated_anomalies
    
    # Filter anomaly points
    anomaly_df = df[df['is_anomaly'] == True]
    
    if len(anomaly_df) == 0:
        return generated_anomalies
    
    # Process each anomaly type
    if 'anomaly_type' in anomaly_df.columns:
        anomaly_types = anomaly_df['anomaly_type'].unique()
        
        for anom_type in anomaly_types:
            # Ensure we handle only our expected types
            if anom_type not in ['contextual', 'pattern', 'point']:
                continue
            
            type_df = anomaly_df[anomaly_df['anomaly_type'] == anom_type]
            
            if anom_type == 'pattern':
                # Special handling for pattern anomalies
                # Group by pattern_group_id to get ranges
                if 'pattern_group_id' in type_df.columns:
                    pattern_groups = type_df.groupby('pattern_group_id')
                    timestamps = []
                    original_values = []
                    
                    for pattern_id, group in pattern_groups:
                        # Get the range of indices for this pattern
                        group_indices = sorted(group.index.tolist())
                        start_idx = group_indices[0]
                        end_idx = group_indices[-1]
                        
                        # Store as range [start_idx, end_idx] for timestamps
                        timestamps.append([start_idx, end_idx])
                        
                        # Extract original values for this pattern (in order)
                        if 'original_value' in group.columns:
                            pattern_original_values = group.loc[group_indices, 'original_value'].tolist()
                        else:
                            # If no original_value column, use current values as placeholder
                            pattern_original_values = group.loc[group_indices, 'value'].tolist()
                        
                        original_values.append(pattern_original_values)
                    
                    # Update the generated_anomalies dict for patterns
                    generated_anomalies[anom_type] = {
                        'count': len(pattern_groups),  # Count of pattern instances, not individual points
                        'timestamps': timestamps,      # List of [start_idx, end_idx] ranges
                        'original_values': original_values  # List of lists, each containing values for one pattern
                    }
                else:
                    # Fallback if pattern_group_id is not available
                    # Treat each point individually (not recommended but handles edge case)
                    timestamps = type_df.index.tolist()
                    
                    if 'original_value' in type_df.columns:
                        original_values = type_df['original_value'].tolist()
                    else:
                        original_values = type_df['value'].tolist()
                    
                    generated_anomalies[anom_type] = {
                        'count': len(timestamps),
                        'timestamps': timestamps,
                        'original_values': original_values
                    }
            
            else:
                # Standard handling for contextual and point anomalies
                timestamps = type_df.index.tolist()
                
                # Extract original values as list (same order as timestamps)
                if 'original_value' in type_df.columns:
                    original_values = type_df['original_value'].tolist()
                else:
                    # If no original_value column, use current values as placeholder
                    original_values = type_df['value'].tolist()
                
                # Update the generated_anomalies dict
                generated_anomalies[anom_type] = {
                    'count': len(timestamps),
                    'timestamps': timestamps,        # List of individual indices
                    'original_values': original_values  # List (same order)
                }
    
    return generated_anomalies

def extract_generated_anomalies_v1(df):
    """
    Extract anomaly information from generated dataframe
    
    Args:
        df: Generated dataframe with columns:
            - 'is_anomaly': boolean indicating if point is anomaly
            - 'anomaly_type': type of anomaly ('contextual', 'pattern', 'point')
            - 'value': current value (potentially modified by anomaly injection)
            - 'original_value': original value before anomaly injection (optional)
    
    Returns:
        dict: Generated anomalies in the format expected by the UI
    """
    generated_anomalies = {
        'contextual': {'count': 0, 'timestamps': [], 'original_values': []},
        'pattern': {'count': 0, 'timestamps': [], 'original_values': []},
        'point': {'count': 0, 'timestamps': [], 'original_values': []}
    }
    
    # Check if anomaly columns exist
    if 'is_anomaly' not in df.columns:
        return generated_anomalies
    
    # Filter anomaly points
    anomaly_df = df[df['is_anomaly'] == True]
    
    if len(anomaly_df) == 0:
        return generated_anomalies
    
    # Process each anomaly type
    if 'anomaly_type' in anomaly_df.columns:
        anomaly_types = anomaly_df['anomaly_type'].unique()
        
        for anom_type in anomaly_types:
            # Ensure we handle only our expected types
            if anom_type not in ['contextual', 'pattern', 'point']:
                continue
                
            type_df = anomaly_df[anomaly_df['anomaly_type'] == anom_type]
            
            # Extract timestamps as list
            timestamps = type_df.index.tolist()
            
            # Extract original values as list (same order as timestamps)
            if 'original_value' in type_df.columns:
                original_values = type_df['original_value'].tolist()
            else:
                # If no original_value column, use current values as placeholder
                original_values = type_df['value'].tolist()
            
            # Update the generated_anomalies dict
            generated_anomalies[anom_type] = {
                'count': len(timestamps),
                'timestamps': timestamps,        # List
                'original_values': original_values  # List (same order)
            }
    
    return generated_anomalies

def calculate_time_span(df):
    if len(df) == 0:
        return 0
    
    try:
        # Try index first
        if pd.api.types.is_datetime64_any_dtype(df.index):
            return (df.index.max() - df.index.min()).days
        
        # Try timestamp column
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            return (timestamps.max() - timestamps.min()).days
        
        # Fallback
        return len(df) // 24 if len(df) > 24 else 1  # Assume hourly data
        
    except (AttributeError, TypeError, ValueError):
        return 0

    def inject_anomalies_into_existing_data(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        df = df.copy()
        np.random.seed(42)  # Optional: For reproducibility
    
        total_points = len(df)
    
        # Contextual Anomalies
        if config['anomalies']['contextual']['enabled']:
            n = int(total_points * config['anomalies']['contextual']['rate'] / 100)
            indices = np.random.choice(df.index, size=n, replace=False)
            df.loc[indices, 'value'] *= config['anomalies']['contextual']['intensity']
            df.loc[indices, 'is_anomaly'] = True
            df.loc[indices, 'anomaly_type'] = 'contextual'
    
        # Pattern Anomalies (simple: flat segments or random spikes)
        if config['anomalies']['pattern']['enabled']:
            n = int(total_points * config['anomalies']['pattern']['rate'] / 100)
            pattern_len = 5  # Inject pattern over a few consecutive points
            for _ in range(n // pattern_len):
                start_idx = np.random.randint(0, total_points - pattern_len)
                pattern_indices = range(start_idx, start_idx + pattern_len)
                df.loc[pattern_indices, 'value'] += np.linspace(0, config['anomalies']['pattern']['intensity'], pattern_len)
                df.loc[pattern_indices, 'is_anomaly'] = True
                df.loc[pattern_indices, 'anomaly_type'] = 'pattern'
    
        # Point Anomalies
        if config['anomalies']['point']['enabled']:
            n = int(total_points * config['anomalies']['point']['rate'] / 100)
            indices = np.random.choice(df.index, size=n, replace=False)
            df.loc[indices, 'value'] *= config['anomalies']['point']['intensity']
            df.loc[indices, 'is_anomaly'] = True
            df.loc[indices, 'anomaly_type'] = 'point'
    
        return df


def display_generated_anomaly_details(df, generated_anomalies, total_anomalies):
    """
    Display detailed breakdown of injected anomalies during data generation
    
    Args:
        df: Generated dataframe
        generated_anomalies: Dict containing info about injected anomalies
                           Format: {
                               'spike_anomalies': {'count': 10, 'timestamps': [...]},
                               'drift_anomalies': {'count': 5, 'timestamps': [...]},
                               'seasonal_anomalies': {'count': 8, 'timestamps': [...]}
                           }
        total_anomalies: Total count of injected anomalies
    """
    
    # Main metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Points", f"{len(df):,}")
    
    with col2:
        col6, col7 = st.columns([0.8, 1])
        with col6:
            st.metric("🚨 Anomalies", f"{total_anomalies:}")
        with col7:
            show_details = st.checkbox(".", value=st.session_state.get('show_generated_anomaly_details', False),help="Check to show detailed breakdown of injected anomalies",key="anomaly_details_checkbox")
            st.session_state.show_generated_anomaly_details = show_details
    
    with col3:
        anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        st.metric("📈 Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col4:
        time_span = calculate_time_span(df)
        st.metric("📅 Time Span", f"{time_span} days")
    
    with col5:
        avg_value = df['value'].mean() if 'value' in df.columns and len(df) > 0 else 0
        st.metric("💹 Avg Value", f"{avg_value:.1f}")
    
    # Detailed anomaly breakdown (shown when button is clicked)
    if st.session_state.get('show_generated_anomaly_details', False) or st.checkbox("Show Injected Anomaly Details", key="show_generated_details_checkbox"):
        
        st.subheader("🔍 Injected Anomaly Analysis")
        
        # Tab-based layout for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 By Anomaly Type", "📅 Timeline View", "📋 Anomaly List", "📈 Generation Stats"])
        
        with tab1:
            # Anomaly breakdown by type
            if generated_anomalies:
                # Create pie chart for anomaly types
                anomaly_types = []
                anomaly_counts = []
                
                for anomaly_type, info in generated_anomalies.items():
                    if info['count'] > 0:
                        # Handle specific anomaly types: contextual, pattern, point
                        if anomaly_type in ['contextual', 'pattern', 'point']:
                            clean_name = anomaly_type.title() + ' Anomalies'
                        else:
                            clean_name = anomaly_type.replace('_', ' ').title()
                        anomaly_types.append(clean_name)
                        anomaly_counts.append(info['count'])
                
                if anomaly_types:
                    fig_pie = px.pie(
                        values=anomaly_counts,
                        names=anomaly_types,
                        title="Injected Anomalies by Type",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Detailed breakdown table
                    breakdown_df = pd.DataFrame([
                        {
                            'Anomaly Type': anomaly_type,
                            'Count Injected': count,
                            'Percentage of Anomalies': f"{(count/total_anomalies)*100:.1f}%" if total_anomalies > 0 else "0%",
                            'Rate in Dataset': f"{(count/len(df))*100:.2f}%" if len(df) > 0 else "0%"
                        }
                        for anomaly_type, count in zip(anomaly_types, anomaly_counts)
                    ])
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No anomalies were injected during generation.")
            else:
                st.info("No anomaly injection information available.")
        
        with tab2:
            # Timeline view of injected anomalies
            if generated_anomalies:
                # Create timeline chart showing when anomalies were injected
                timeline_data = []
                colors = px.colors.qualitative.Set1

                for i, (anomaly_type, info) in enumerate(generated_anomalies.items()):
                    if info['count'] > 0 and 'timestamps' in info:
                        # Handle specific anomaly types
                        if anomaly_type in ['contextual', 'pattern', 'point']:
                            clean_name = anomaly_type.title() + ' Anomalies'
                        else:
                            clean_name = anomaly_type.replace('_', ' ').title()

                        # Handle different timestamp formats based on anomaly type
                        if anomaly_type == 'pattern':
                            # Pattern anomalies have timestamps as ranges [[start, end], [start, end], ...]
                            for j, timestamp_range in enumerate(info['timestamps']):
                                if isinstance(timestamp_range, list) and len(timestamp_range) == 2:
                                    start_idx, end_idx = timestamp_range
                                    # Add all points in the pattern range
                                    for timestamp in range(start_idx, end_idx + 1):
                                        if timestamp in df.index:
                                            timeline_data.append({
                                                'Timestamp': timestamp,
                                                'Type': clean_name,
                                                'Value': df.loc[timestamp, 'value'],
                                                'Color': colors[i % len(colors)],
                                                'Pattern_ID': j,  # Track which pattern instance
                                                'Position_in_Pattern': timestamp - start_idx  # Position within pattern
                                            })
                                else:
                                    # Fallback for single timestamp (shouldn't happen with new format)
                                    if timestamp_range in df.index:
                                        timeline_data.append({
                                            'Timestamp': timestamp_range,
                                            'Type': clean_name,
                                            'Value': df.loc[timestamp_range, 'value'],
                                            'Color': colors[i % len(colors)],
                                            'Pattern_ID': j,
                                            'Position_in_Pattern': 0
                                     })
                        else:
                            # Contextual and point anomalies have individual timestamps
                            for timestamp in info['timestamps']:
                                if timestamp in df.index:
                                    timeline_data.append({
                                        'Timestamp': timestamp,
                                        'Type': clean_name,
                                        'Value': df.loc[timestamp, 'value'],
                                        'Color': colors[i % len(colors)],
                                        'Pattern_ID': None,  # Not applicable for individual anomalies
                                        'Position_in_Pattern': None
                                    })

                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)

                    # Create scatter plot showing anomalies over time
                    fig_timeline = px.scatter(
                        timeline_df,
                        x='Timestamp',
                        y='Value',
                        color='Type',
                        title="Injected Anomalies Timeline",
                        hover_data={
                            'Type': True, 
                            'Value': ':.2f',
                            'Pattern_ID': True,
                            'Position_in_Pattern': True
                        },
                        size_max=10
                    )

                    # Add the main data line
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['value'],
                            mode='lines',
                            name='Generated Data',
                            line=dict(color='lightgray', width=1),
                            opacity=0.6
                        )
                    )

                    # Add pattern range indicators (optional - visual enhancement)
                    if 'pattern' in generated_anomalies and generated_anomalies['pattern']['count'] > 0:
                        pattern_data = timeline_df[timeline_df['Type'] == 'Pattern Anomalies']
                        if not pattern_data.empty:
                            for pattern_id in pattern_data['Pattern_ID'].unique():
                                if pd.notna(pattern_id):
                                    pattern_points = pattern_data[pattern_data['Pattern_ID'] == pattern_id]
                                    if len(pattern_points) > 1:
                                        # Add a line connecting pattern points to show the range
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=pattern_points['Timestamp'],
                                                y=pattern_points['Value'],
                                                mode='lines',
                                                name=f'Pattern {int(pattern_id)} Range',
                                                line=dict(color='red', width=2, dash='dot'),
                                                opacity=0.7,
                                                showlegend=True if pattern_id == 0 else False  # Only show legend for first pattern
                                            )
                                        )

                    fig_timeline.update_layout(height=500)
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Enhanced statistics display
                    st.subheader("Anomaly Statistics")
                    col1, col2, col3 = st.columns(3)
            
                    with col1:
                        st.metric("Total Anomaly Instances", 
                                 sum(info['count'] for info in generated_anomalies.values()))
            
                    with col2:
                        total_anomalous_points = len(timeline_df)
                        st.metric("Total Anomalous Data Points", total_anomalous_points)
            
                    with col3:
                        anomaly_percentage = (total_anomalous_points / len(df)) * 100
                        st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

                    # Detailed breakdown by type
                    st.subheader("Anomaly Breakdown")
                    breakdown_data = []
                    for anomaly_type, info in generated_anomalies.items():
                        if info['count'] > 0:
                            if anomaly_type == 'pattern':
                                # For patterns, count both instances and points
                                total_points = sum(len(values) for values in info['original_values'])
                                breakdown_data.append({
                                    'Type': anomaly_type.title(),
                                    'Instances': info['count'],
                                    'Data Points': total_points,
                                    'Avg Points per Instance': total_points / info['count'] if info['count'] > 0 else 0
                                })
                            else:
                                # For individual anomalies
                                breakdown_data.append({
                                    'Type': anomaly_type.title(),
                                    'Instances': info['count'],
                                    'Data Points': info['count'],
                                    'Avg Points per Instance': 1.0
                                })
            
                    if breakdown_data:
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True)

                    # Hourly distribution (updated to handle pattern ranges)
                    if len(timeline_data) > 0:
                        st.subheader("Anomaly Distribution by Hour of Day")
                
                        # Create hourly distribution data
                        hourly_data = []
                        for row in timeline_data:
                            # Convert timestamp to hour (assuming timestamp is an index or time-based)
                            if isinstance(row['Timestamp'], (int, float)):
                                # If timestamp is numeric index, map to hour of day
                                # This assumes your dataframe has a datetime index or time column
                                if hasattr(df.index, 'hour'):
                                    hour = df.index[int(row['Timestamp'])].hour if int(row['Timestamp']) < len(df) else 0
                                else:
                                    # Fallback: distribute across 24 hours based on position
                                    hour = int(row['Timestamp']) % 24
                            else:
                                # If timestamp is already datetime
                                hour = pd.to_datetime(row['Timestamp']).hour
                    
                            hourly_data.append({
                                'Hour': hour,
                                'Type': row['Type'],
                                'Count': 1
                            })
                
                        if hourly_data:
                            hourly_df = pd.DataFrame(hourly_data)
                            hourly_counts = hourly_df.groupby(['Hour', 'Type']).size().reset_index(name='Count')

                            fig_hourly = px.bar(
                                hourly_counts,
                                x='Hour',
                                y='Count',
                                color='Type',
                                title="Anomaly Distribution by Hour of Day",
                                labels={'Hour': 'Hour of Day', 'Count': 'Number of Anomalous Data Points'}
                            )
                            #fig_hourly.update_xaxis(tickmode='linear', tick0=0, dtick=1)
                            fig_hourly.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                            st.plotly_chart(fig_hourly, use_container_width=True)

                    # Pattern-specific analysis (new section)
                    if 'pattern' in generated_anomalies and generated_anomalies['pattern']['count'] > 0:
                        st.subheader("Pattern Anomaly Analysis")
                
                        pattern_info = generated_anomalies['pattern']
                        pattern_analysis = []
                
                        for i, (timestamp_range, original_values) in enumerate(zip(pattern_info['timestamps'], pattern_info['original_values'])):
                            if isinstance(timestamp_range, list) and len(timestamp_range) == 2:
                                start_idx, end_idx = timestamp_range
                                pattern_length = end_idx - start_idx + 1
                        
                                pattern_analysis.append({
                                    'Pattern ID': i,
                                    'Start Index': start_idx,
                                    'End Index': end_idx,
                                    'Length': pattern_length,
                                    'Start Value': original_values[0] if original_values else 'N/A',
                                    'End Value': original_values[-1] if original_values else 'N/A',
                                    'Range': f"[{start_idx}-{end_idx}]"
                                })
                
                        if pattern_analysis:
                            pattern_df = pd.DataFrame(pattern_analysis)
                            st.dataframe(pattern_df, use_container_width=True)

                else:
                    st.info("No timeline data available for injected anomalies.")
            else:
                st.info("No timeline information available.")

        with tab3:
            # Detailed list of injected anomalies
            if generated_anomalies:
                # Filter by anomaly type
                available_types = []
                for k, v in generated_anomalies.items():
                    if v['count'] > 0:
                        if k in ['contextual', 'pattern', 'point']:
                            available_types.append(k.title() + ' Anomalies')
                        else:
                            available_types.append(k.replace('_', ' ').title())
                
                if available_types:
                    selected_type = st.selectbox(
                        "Select Anomaly Type to View Details:",
                        options=available_types
                    )
                    
                    # Find corresponding data
                    original_type = None
                    for k in generated_anomalies.keys():
                        if k in ['contextual', 'pattern', 'point']:
                            display_name = k.title() + ' Anomalies'
                        else:
                            display_name = k.replace('_', ' ').title()
                            
                        if display_name == selected_type:
                            original_type = k
                            break
                    
                    if original_type and 'timestamps' in generated_anomalies[original_type]:
                        timestamps = generated_anomalies[original_type]['timestamps']
                        original_values = generated_anomalies[original_type].get('original_values', [])
                        print(f"original_type:{original_type} - timestamps{timestamps}, original_values{original_values}")

                        # Create detailed list
                        anomaly_details = []
                        for i, ts in enumerate(timestamps):
                            print(f"Index and ts are : index:{i} ts:{ts}")

                            # Q1 Fix: Handle pattern-based anomalies (ranges) vs individual timestamps
                            if original_type == 'pattern':
                                # ts is a list [start_idx, end_idx] for patterns
                                if isinstance(ts, list) and len(ts) == 2:
                                    start_idx, end_idx = ts
                                    # Check if both start and end indices exist in dataframe
                                    if start_idx in df.index and end_idx in df.index:
                                        try:
                                            # Q2 Fix: For pattern case, original_values[i] is a list of values
                                            if isinstance(original_values, list) and i < len(original_values):
                                                original_val = original_values[i]  # This is a list for patterns
                                            else:
                                                original_val = 'N/A'
                                        except (IndexError, KeyError):
                                            original_val = 'N/A'

                                        # E4: Create flattened values for pattern anomalies
                                        flatten_timestamps = list(range(start_idx, end_idx + 1))
                                        if isinstance(original_val, list):
                                            flatten_original_values = original_val
                                        else:
                                            flatten_original_values = ['N/A'] * len(flatten_timestamps)

                                        # Get actual datetime info from dataframe index
                                        if 'timestamp' in df.columns:
                                        
                                            start_ts_str = df.loc[start_idx, 'timestamp']
                                            end_ts_str = df.loc[end_idx, 'timestamp']

                                            # 2. Convert these to datetime objects
                                            start_datetime = pd.to_datetime(start_ts_str)
                                            end_datetime = pd.to_datetime(end_ts_str)

                                            # 3. Now you can safely extract the day and hour
                                            start_day = start_datetime.strftime('%A')
                                            end_day = end_datetime.strftime('%A')
                                            start_hour = start_datetime.hour
                                            end_hour = end_datetime.hour

 
                                        anomaly_details.append({
                                            'Timestamp': f"{start_ts_str} to {end_ts_str}",  # Q3 Fix: Keep as list [start_idx, end_idx] for patterns
                                            'Anomaly Type': selected_type,
                                            'Day of Week': f"{start_day} to {end_day}",
                                            'Hour': f"{start_hour} to {end_hour}",
                                            'Original Value': original_val,
                                            'Flatten Timestamp': flatten_timestamps,  # E4: Flattened individual timestamps
                                            'Flatten Original Value': flatten_original_values,  # E4: Flattened individual original values
                                            'Pattern Length': len(flatten_timestamps)
                                        })
                                else:
                                    print(f"Warning: Invalid pattern timestamp format: {ts}")
                            else:
                                # Handle contextual and point anomalies (individual timestamps)
                                if ts in df.index:
                                    try:
                                        if isinstance(original_values, list):
                                            original_val = original_values[i] if i < len(original_values) else 'N/A'
                                        elif isinstance(original_values, dict):
                                            original_val = original_values.get(ts, 'N/A')
                                        else:
                                            original_val = 'N/A'
                                    except (IndexError, KeyError):
                                        original_val = 'N/A'
 
                                    # E4: For individual anomalies, flatten values are same as original values
                                    flatten_timestamps = [ts]
                                    flatten_original_values = [original_val]

                                    # Get actual datetime info from dataframe index
                                    ts_datetime = df.index[ts] if hasattr(df.index, 'strftime') else pd.to_datetime(df.index[ts]) if ts in df.index else None

                                    ts_str = df.loc[ts, 'timestamp']
                                    ts_datetime = pd.to_datetime(ts_str)
                                    day = ts_datetime.strftime('%A')
                                    hour = ts_datetime.hour

 
                                    anomaly_details.append({
                                        'Timestamp': ts_str,
                                        'Anomaly Type': selected_type,
                                        'Day of Week': day,
                                        'Hour': hour,
                                        'Original Value': original_val,
                                        'Flatten Timestamp': flatten_timestamps,  # E4: Same as Timestamp for individual anomalies
                                        'Flatten Original Value': flatten_original_values,  # E4: Same as Original Value for individual anomalies
                                        'Pattern Length': 1  # Individual anomalies have length 1
                                    })
 
                        if anomaly_details:
                            anomaly_df = pd.DataFrame(anomaly_details)
         
                            # Display main anomaly information (hide flatten columns for cleaner view)
                            display_columns = ['Timestamp', 'Anomaly Type', 'Day of Week', 'Hour', 'Original Value', 'Pattern Length']
                            st.dataframe(anomaly_df[display_columns], use_container_width=True, hide_index=True)
         
                            # Option to show detailed flattened data
                            if st.checkbox("Show Flattened Data for Metrics Evaluation"):
                                st.subheader("Flattened Anomaly Data (for Metrics)")
             
                                # Create flattened dataframe for metrics evaluation
                                flattened_data = []
                                for _, row in anomaly_df.iterrows():
                                    flatten_ts = row['Flatten Timestamp']
                                    flatten_orig = row['Flatten Original Value']
                 
                                    for j, (ts, orig_val, inj_val) in enumerate(zip(flatten_ts, flatten_orig, flatten_inj)):
                                        flattened_data.append({
                                           'Individual Timestamp': ts,
                                           'Individual Original Value': orig_val,
                                           'Anomaly Type': row['Anomaly Type'],
                                           'Parent Anomaly': f"{row['Anomaly Type']}_{anomaly_df.index[_]}",
                                           'Position in Pattern': j if row['Anomaly Type'] == 'pattern' else 0
                                        })
             
                                if flattened_data:
                                    flattened_df = pd.DataFrame(flattened_data)
                                    st.dataframe(flattened_df, use_container_width=True, hide_index=True)
                 
                                    # Metrics summary
                                    total_individual_points = len(flattened_df)
                                    pattern_points = len(flattened_df[flattened_df['Anomaly Type'] == 'pattern'])
                                    individual_points = total_individual_points - pattern_points
                 
                                    st.info(f"**Metrics Summary**: {total_individual_points} total anomalous data points "
                                        f"({pattern_points} from patterns, {individual_points} individual)")
 
                            # Download option
                            csv = anomaly_df.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {selected_type} Anomalies CSV",
                                data=csv,
                                file_name=f"generated_{selected_type.lower().replace(' ', '_')}_anomalies.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No detailed data available for this anomaly type.")
                    else:
                        st.info("No timestamp data available for selected type.")
                else:
                    st.info("No anomaly types with data available.")
            else:
                st.info("No anomaly list available.")
        
        with tab4:
            # Generation statistics
            if generated_anomalies:
                st.subheader("📊 Generation Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most/least common anomaly types
                    counts = {k: v['count'] for k, v in generated_anomalies.items() if v['count'] > 0}
                    
                    if counts:
                        most_common = max(counts.items(), key=lambda x: x[1])
                        least_common = min(counts.items(), key=lambda x: x[1])
                        
                        st.metric(
                            "🔥 Most Injected Type", 
                            most_common[0].title() + (' Anomalies' if most_common[0] in ['contextual', 'pattern', 'point'] else ''),
                            delta=f"{most_common[1]} instances"
                        )
                        
                        if len(counts) > 1:
                            st.metric(
                                "❄️ Least Injected Type", 
                                least_common[0].title() + (' Anomalies' if least_common[0] in ['contextual', 'pattern', 'point'] else ''),
                                delta=f"{least_common[1]} instances"
                            )
                
                with col2:
                    # Time-based statistics
                    all_timestamps = []
                    all_timestamps = get_all_anomalous_timestamps(generated_anomalies)
                    print(f"all_timestamps is : {all_timestamps}")
                    
                    if all_timestamps:
                        # Convert to datetime for analysis
                        dt_timestamps = [pd.to_datetime(ts) for ts in all_timestamps]
                        
                        # Most common hour
                        hours = [ts.hour for ts in dt_timestamps]
                        print(f"hours is : {hours}")
                        from collections import Counter

                        hours_list = hours.tolist() if hasattr(hours, 'tolist') else list(hours)
                        print(f"hours_list is : {hours_list}")

                        if hours_list:
                            # Use Counter to find the most common element directly
                            most_common_hour = Counter(hours_list).most_common(1)[0][0]
                        else:
                            most_common_hour = 0

                        print(f" most common hour is : {most_common_hour}")

                        
                        # Most common day
                        days = [ts.strftime('%A') for ts in dt_timestamps]
                        most_common_day = max(set(days), key=days.count) if days else 'N/A'
                        
                        st.metric("🕐 Peak Injection Hour", f"{most_common_hour:02d}:00")
                        st.metric("📅 Peak Injection Day", most_common_day)
                
                # Injection density analysis
                if len(all_timestamps) > 0:
                    st.subheader("📈 Injection Density")
                    
                    # Calculate injection density over time
                    dt_timestamps = pd.to_datetime(all_timestamps)
                    injection_series = pd.Series(1, index=dt_timestamps)
                    
                    # Resample to show density
                    if len(dt_timestamps) > 24:  # Use daily if long time series
                        density = injection_series.resample('D').sum()
                        freq_label = "Daily"
                    else:  # Use hourly for shorter series
                        density = injection_series.resample('H').sum()
                        freq_label = "Hourly"
                    
                    fig_density = px.bar(
                        x=density.index,
                        y=density.values,
                        title=f"{freq_label} Anomaly Injection Density",
                        labels={'x': 'Time', 'y': 'Number of Injected Anomalies'}
                    )
                    st.plotly_chart(fig_density, use_container_width=True)
            else:
                st.info("No generation statistics available.")

def display_sidebar_for_data_builder():
    studio = ComprehensiveAnomalyStudio()
    # Sidebar for SOURCE configuration
    with st.sidebar:
        st.header("⏳ Chronosync Generator ")
            
        # Date range
        start_date = st.date_input("Start Date", datetime(2024, 1, 1).date())
        end_date = st.date_input("End Date", datetime(2024, 2, 1).date())
            
        # Data interval with tooltip
        interval_minutes = st.selectbox(
            "Data Interval (minutes)", 
            [15, 30, 60], 
            index=0,
            help="Time between consecutive data points. 15min = 96 points/day, 60min = 24 points/day"
        )
            
        st.subheader("⏰ Time Categories")
            
        # Peak hours with improved interface
        peak_hours_str = st.text_input(
            "Peak Hours (0-23)", 
            "8-10,17-19",
            help="Format: '8-10,17-19' for morning and evening peaks. Use ranges (8-10) or individual hours (8,9,10)"
        )
        peak_hours = studio.parse_time_ranges(peak_hours_str)
        if peak_hours:
            st.info(f"Peak hours: {peak_hours}")
            
        # Off hours with improved interface
        off_hours_str = st.text_input(
            "Off Hours (0-23)", 
            "0-5,22-23",
            help="Format: '0-5,22-23' for night hours. These are typically low-activity periods"
        )
        off_hours = studio.parse_time_ranges(off_hours_str)
        if off_hours:
            st.info(f"Off hours: {off_hours}")
            
        st.header("⚖️  Normality Calibrator ")
            
        base_value = st.number_input(
            "Base Value", 
            min_value=1.0, 
            value=100.0, 
            step=1.0,
            help="Starting point for all calculations. This represents the 'normal' value"
        )
            
        noise_level = st.slider(
            "Noise Level", 
            0.0, 20.0, 5.0, 0.5,
            help="Random variation added to data. Higher values = more realistic but noisier data"
        )
            
        st.header("🎶 Cyclical Pattern Generator")
            
        daily_seasonality = st.checkbox("Daily Seasonality", True, help="Predictable daily pattern (hour by hour). Causes values to rise/fall during different hours)")
        weekly_seasonality = st.checkbox("Weekly Seasonality", True, help="Predictable weekly pattern (day by day). Changes values depending on day of week")  
        monthly_seasonality = st.checkbox("Monthly Seasonality", False, help="Predictable monthly pattern (month by month). Values shift depending on month")
            
        seasonality_strength = {}
        if daily_seasonality:
            seasonality_strength['daily'] = st.slider(
                "Daily Strength", 0.0, 50.0, 20.0, 1.0,
                help="How much values vary throughout the day i.e How strong or noticeable these patterns are. Higher = bigger day/night differences i.e Higher strength = bigger ups and downs . "
            )
        if weekly_seasonality:
            seasonality_strength['weekly'] = st.slider(
                "Weekly Strength", 0.0, 30.0, 15.0, 1.0,
                help="How much values vary throughout the week. Higher = bigger weekday/weekend differences"
            )
        if monthly_seasonality:
            seasonality_strength['monthly'] = st.slider(
                "Monthly Strength", 0.0, 20.0, 10.0, 1.0,
                help="How much values vary throughout the month. Higher = bigger month start/end differences"
            )
    
def date_time_configuration_for_data():
    studio = ComprehensiveAnomalyStudio()
    with st.expander(" ⏳ Chronosync Generator", expanded=False):
        st.markdown("<h3 style='color:#0072c6;'> ⏳ Chronosync Generator </h3>", unsafe_allow_html=True)
        # Date range
        start_date = st.date_input("Start Date", datetime(2024, 1, 1).date())
        end_date = st.date_input("End Date", datetime(2024, 2, 1).date())

        # Data interval with tooltip
        interval_minutes = st.selectbox(
            "Data Interval (minutes)", 
            [15, 30, 60], 
            index=0,
            help="Time between consecutive data points. 15min = 96 points/day, 60min = 24 points/day"
        )
        st.subheader("⏰ Time Categories")

        # Peak hours with improved interface
        peak_hours_str = st.text_input(
            "Peak Hours (0-23)",
            "8-10,17-19",
            help="Format: '8-10,17-19' for morning and evening peaks. Use ranges (8-10) or individual hours (8,9,10)"
        )
        peak_hours = studio.parse_time_ranges(peak_hours_str)
        if peak_hours:
            st.info(f"Peak hours: {peak_hours}")

        # Off hours with improved interface
        off_hours_str = st.text_input(
            "Off Hours (0-23)",
            "0-5,22-23",
            help="Format: '0-5,22-23' for night hours. These are typically low-activity periods"
        )
        off_hours = studio.parse_time_ranges(off_hours_str)
        if off_hours:
            st.info(f"Off hours: {off_hours}")

        return start_date, end_date, interval_minutes, peak_hours, off_hours

def base_values_configuration_for_data():
    with st.expander(" ⚖️  Normality Calibrator ", expanded=False):
        st.markdown("<h3 style='color:#0072c6;'> ⚖️  Normality Calibrator ⚖️  </h3>", unsafe_allow_html=True)

        with st.expander("🔍 Details & Explanation"):
            st.markdown("""
                - **Base Value:**
                This is the starting or “normal” value of your data — the typical number you expect when everything is stable and no special patterns exist. Think of it as the baseline around which your data fluctuates.
                - **Noise Level:**
                This adds some randomness to your data, mimicking small unpredictable fluctuations or measurement errors that happen naturally in real-world data. Higher noise means more “random wiggles” around the base value.
            """)

        base_value = st.number_input(
            "Base Value",
            min_value=1.0,
            value=100.0,
            step=1.0,
            help="Starting point for all calculations. This represents the 'normal' value"
        )

        noise_level = st.slider(
            "Noise Level",
            0.0, 20.0, 5.0, 0.5,
            help="Random variation added to data. Higher values = more realistic but noisier data"
        )
        return base_value, noise_level

def seasonality_configuration_for_data():
    with st.expander("🔄  Rhythmics Engine", expanded=False):
        st.markdown("<h3 style='color:#0072c6;'> 🔄  Rhythmics Engine</h3>", unsafe_allow_html=True)

        with st.expander("🔍 Details & Explanation"):
            st.markdown("""
                Seasonality is about repeating patterns that happen regularly and predictably over time.
                - **Example** :
                - **Daily seasonality**: Your data tends to rise and fall in a predictable way every day. Like electricity usage peaking around 6 PM every evening.
                - **Weekly seasonality**: Your data changes on a weekly rhythm. Maybe traffic is heavier on Mondays and lighter on Sundays.
                - **Monthly seasonality**: Your data shifts over the course of the month, e.g., sales increase at the end of the month.
                <br>
                The strengths you set control how big these repeating ups and downs are — stronger strength means more variation.
                These are smooth, sinusoidal waves that add or subtract a certain amount to/from your base value.
                Strength: Strength controls magnitude of repeating fluctuations. Imagine the waves on the ocean: strength controls how tall or deep each wave is, but the waves keep coming regularly.
            """)

        daily_seasonality = st.checkbox("Daily Seasonality", True, help="Predictable daily pattern (hour by hour). Causes values to rise/fall during different hours")
        weekly_seasonality = st.checkbox("Weekly Seasonality", True, help="Predictable weekly pattern (day by day). Changes values depending on day of week")
        monthly_seasonality = st.checkbox("Monthly Seasonality", True, help="Predictable monthly pattern (month by month). Values shift depending on month")

        seasonality_strength = {}
        if daily_seasonality:
            seasonality_strength['daily'] = st.slider(
                "Daily Strength", 0.0, 50.0, 20.0, 1.0,
                help="How much values vary throughout the day i.e How strong or noticeable these patterns are. Higher = bigger day/night differences i.e Higher strength = bigger ups and downs . "
            )
        if weekly_seasonality:
            seasonality_strength['weekly'] = st.slider(
                "Weekly Strength", 0.0, 30.0, 15.0, 1.0,
                help="How much values vary throughout the week. Higher = bigger weekday/weekend differences"
            )
        if monthly_seasonality:
            seasonality_strength['monthly'] = st.slider(
                "Monthly Strength", 0.0, 20.0, 10.0, 1.0,
                help="How much values vary throughout the month. Higher = bigger month start/end differences"
            )

        return daily_seasonality, weekly_seasonality, monthly_seasonality, seasonality_strength

def trend_config_for_data():
    with st.expander("📈 Trend Multiplier Controls", expanded=False):
        st.markdown("<h3 style='color:#0072c6;'> 📈 Trend Multiplier Controls</h3>", unsafe_allow_html=True)

        #st.subheader("📈 Trend Configuration")
        st.markdown("*Multipliers that affect the base value in different time contexts*")

        with st.expander("🔍 Details & Explanation"):
            st.markdown("""
                Trend multipliers scale the entire value up or down for specific contexts or special times.
                These multipliers act as multiplicative factors, adjusting the overall level of your data in certain conditions.
                - **Example:**
                - **Weekend multiplier:** On weekends, the entire value might be multiplied by 0.8 (20% lower) because your business is slower.
                - **Peak hour multiplier:** During peak hours, the value might be multiplied by 1.5 (50% higher) to reflect busier times.
                - **Holiday multiplier:** On holidays, maybe the multiplier is 0.5 because the activity drops a lot.
                - **Weekday-specific multipliers:** Each day of the week can have its own multiplier. Monday could be 1.2, Tuesday 1.0, etc., to reflect day-specific trends.
                <br>
                These are more like “adjustments” on top of your base + seasonality, applied at certain times.
            """)


            
        trends = {}
        trends['weekend'] = st.slider(
            "Weekend Multiplier", 0.1, 2.0, 0.8, 0.1, 
            help="Multiply base value on weekends. <1.0 = lower, >1.0 = higher than weekdays"
        )
        trends['peak_hour'] = st.slider(
            "Peak Hour Multiplier", 0.5, 3.0, 1.5, 0.1, 
            help="Multiply base value during peak hours. Usually >1.0 for higher activity"
        )
        trends['off_hour'] = st.slider(
            "Off Hour Multiplier", 0.1, 1.5, 0.7, 0.1, 
            help="Multiply base value during off hours. Usually <1.0 for lower activity"
        )
        trends['holiday'] = st.slider(
            "Holiday Multiplier", 0.1, 1.5, 0.6, 0.1, 
            help="Multiply base value on holidays. Usually <1.0 for reduced activity"
        )
            
        st.subheader("📅 Weekday Specific Trends")
        st.markdown("*Different multipliers for each day of the week*")
        
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_trends = []
            
        for i, day_name in enumerate(weekday_names):
            default_val = 0.8 if i >= 5 else 1.2  # Weekend vs weekday default
            weekday_trends.append(st.slider(
                f"{day_name}", 0.1, 2.0, default_val, 0.1, 
                help=f"Multiplier for {day_name}. Allows fine-tuning weekly patterns"
            ))
            
        trends['weekdays'] = weekday_trends
        return trends

def anomaly_config_for_data():
    with st.expander("🚨 Outlier Injector ", expanded=False):
        st.markdown("<h3 style='color:#0072c6;'>🚨 Outlier Injector </h3>", unsafe_allow_html=True)

        #st.subheader("💉 Outlier Injector ")
        st.markdown("*Configure different types of synthetic anomalies*")
            
        with st.expander("🔍 Details & Explanation"):
            st.markdown("""
                - **1. Contextual Anomalies:**
                - **Definition:** These occur when a data point is unusual given the context or environment it appears in.
                - **Example:** A temperature of 30°C in summer might be normal but the same 30°C in winter might be an anomaly.
                - **Impact:** The anomaly depends on time, season, or other external factors.
                - **Use case:** Detecting unusual behavior compared to expected patterns based on time or other contextual features.

                - **2. Pattern-Based Anomalies:**
                - **Definition:** These anomalies are detected when the overall pattern or shape of a sequence changes unexpectedly.
                - **Example:** A sudden shift from a steady increasing trend to a flat or decreasing trend in sensor readings.
                - **Impact:** Affects a series of points rather than a single data point.
                - **Use case:** Finding unusual trends or changes in repetitive sequences.

                - **3. Point-Based Anomalies:**
                - **Definition:** These are isolated individual points that stand out significantly from their neighbors or the rest of the data.
                - **Example:** A single spike in network traffic or a sudden drop in temperature.
                - **Impact:** These anomalies are local and affect one or very few points.
                - **Use case:** Identifying sudden spikes or dips.

                - **How They Impact the Data**
                Contextual anomalies depend on comparing a point to its expected context (time, environment).
                Pattern anomalies influence sequences and change the shape of the data over time.
                Point anomalies are sudden, isolated deviations.

                - **What Does "Intensity" Mean?**
                - **Definition:** Intensity controls how strong or pronounced the anomaly is compared to normal data values.
                - **Effect:**
                A high intensity anomaly causes a large deviation from normal values (e.g., a big spike or drop).
                A low intensity anomaly causes a subtle deviation that may be harder to detect.
                For example, if your normal value is 100:
                At intensity = 1, an anomaly might be a small increase to 110.
                At intensity = 5, the anomaly might jump to 150 or more, clearly standing out.
            """)
        anomalies = {}
            
        # Contextual anomalies
        with st.expander("🎯 Contextual Anomalies", expanded=True):
            anomalies['contextual'] = {
                'enabled': st.checkbox("Enable Contextual Anomalies", True),
                'rate': st.slider(
                    "Contextual Anomaly Rate (%)", 0.0, 5.0, 1.0, 0.1, 
                    help="Percentage of data points to make contextually anomalous"
                ),
                'intensity': st.slider(
                    "Contextual Intensity", 0.5, 3.0, 1.5, 0.1, 
                    help="How much to deviate from expected context. Higher = more obvious anomalies"
                )
            }
            
        # Pattern anomalies
        with st.expander("📊 Pattern Anomalies", expanded=True):
            anomalies['pattern'] = {
                'enabled': st.checkbox("Enable Pattern Anomalies", True),
                'rate': st.slider(
                    "Pattern Anomaly Rate (%)", 0.0, 3.0, 0.5, 0.1, 
                    help="Percentage of data to include in pattern-breaking sequences"
                ),
                'intensity': st.slider(
                    "Pattern Intensity", 0.5, 2.0, 1.3, 0.1, 
                    help="How much patterns deviate from expected sequences"
                )
            }
        
        # Point anomalies  
        with st.expander("📍 Point Anomalies", expanded=True):
            anomalies['point'] = {
                'enabled': st.checkbox("Enable Point Anomalies", True),
                'rate': st.slider(
                    "Point Anomaly Rate (%)", 0.0, 3.0, 0.8, 0.1, 
                    help="Percentage of individual data points that are statistical outliers"
                ),
                'intensity': st.slider(
                    "Point Intensity", 1.5, 5.0, 2.5, 0.1, 
                    help="How far point anomalies deviate from normal (multiplier). Higher = more extreme outliers"
                )
            }
        return anomalies
        
def chech_arrow_compatibility(df):

    import pyarrow as pa
    st.write("🔍 Checking Arrow compatibility:")

    for col in df.columns:
        try:
            pa.array(df[col])
        except Exception as e:
            st.error(f"❌ Column `{col}` failed: {e}")


def main():
    st.markdown("""
        <style>
        .stButton>button {
            border: 2px solid lightblue;
        }
        .stTextInput>div>div>input {
            border: 2px solid lightblue;
        }
        .stTextArea>div>div>textarea {
            border: 2px solid lightblue;
        }
        .stSelectbox>div {
            border: 2px solid lightblue;
        }
        .stCheckbox>label {
            border: 2px solid lightblue;
            padding: 5px;
            border-radius: 5px;
        }
        .stRadio>label {
            border: 2px solid lightblue;
            padding: 5px;
            border-radius: 5px;
        }
        .stSlider {
            border: 2px solid lightblue;
            padding: 10px;
            border-radius: 5px;
        }
        .stFileUploader>section {
            border: 2px solid lightblue;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # for Header spacing 
    st.markdown(""" <style> [data-testid="stAppViewContainer"] { padding-top: 0rem !important; margin-top: 0rem !important; } [data-testid="stMainContent"] { padding-top: 0rem !important; } h1 { margin-top: 0rem !important; } </style> """, unsafe_allow_html=True)

    # Add this CSS to make tabs look like proper tabs
    st.markdown(""" <style> .stTabs [data-baseweb="tab-list"] { gap: 2px; } .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-left: 20px; padding-right: 20px; } .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff6b6b; } </style> """, unsafe_allow_html=True)

    #st.title("🎯 AnomalyExperiments Lab ")
    #st.markdown("From Synthetic Data to Smart Detection - The Complete Anomaly Analysis Pipeline")


    st.markdown("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
            <h1 style="color: #0072c6; font-weight: 700; font-size: 3.0rem; margin-bottom: 0.25em;">
                📊 ➡️  🕵️  ➡️  ⚖️  Anomlytics
            </h1>
            <p style="color: #555555; font-size: 1.5rem; font-weight: 500; margin-top: 0;">
                Anomlytics: The A.I. Pipeline for Synthetic Data & Anomaly Detection
            </p>
        </div>
    """, unsafe_allow_html=True)


    # Add space before tabs
    st.write("")  # Single line break
    st.write("")  # Double line break for more space

    studio = ComprehensiveAnomalyStudio()
    
    # Main tabs
    # tab1, tab2, tab3, tab4 = st.tabs(["📊 GENERATE: Synthetic Data Generation", "🔍 ANALYZE: Anomaly Detection", "📈 EVALUATE: Evaluate Results", "📚 Help & Glossary"])
    # Inject custom CSS

    st.markdown("""
        <style>
        /* All tab labels */
        .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    /* Active tab */
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #007BFF;  /* Change this line */
    }
    </style>
    """, unsafe_allow_html=True)





    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 SYNTHESIZE: Data Creation", 
        "🕵️ DETECT: Anomaly Analysis", 
        "⚖️  EVALUATE: Performance Comparison", 
        "📚 GUIDE: Concepts & Glossary"
    ])


    
    with tab1:
        st.header("⚙️ Data Synthesis Configuration")

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 2, 1).date()
        interval_minutes = 15
        peak_hours = []
        off_hours = []
        base_value = 100
        noise_level = 5
        daily_seasonality = False
        weekly_seasonality = False
        monthly_seasonality = False
        seasonality_strength = {}

        with st.expander("🔎 Trend & Seasonality Primer"):
            st.markdown("""
                Here’s a detailed explanation to help you understand the difference between Seasonality and Trend Multipliers:

                - **Seasonality** captures predictable repeating patterns like daily peaks or monthly cycles.
                - **Trend Multipliers** adjust the overall data level for special periods like weekends or holidays.

                Think of seasonality as the “rhythm” of your data, while trend multipliers are the “volume control” during different contexts.
                Set seasonality strengths to control how much the data fluctuates regularly through day/week/month.
                Set trend multipliers to control how much the data level changes during specific time contexts (e.g., weekends or holidays).
            """)

        col3, col4, col5 = st.columns([1, 1, 1])
        with col3:
            start_date, end_date, interval_minutes, peak_hours, off_hours = date_time_configuration_for_data()
        with col4:
            base_value, noise_level = base_values_configuration_for_data()
        with col5:
            daily_seasonality, weekly_seasonality, monthly_seasonality, seasonality_strength = seasonality_configuration_for_data()
        
        # Main content for SOURCE
        col1, col2 = st.columns([1, 1])
        trends = {}
        with col1:
            trends = trend_config_for_data()

        anomalies = {}
        with col2:
            anomalies = anomaly_config_for_data()

        # Generate data button
        st.header("🚀 Data Generation")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🎲 Generate Synthetic Data", type="primary", use_container_width=True):
                if start_date >= end_date:
                    st.error("❌ End date must be after start date!")
                    return
                
                if not peak_hours and not off_hours:
                    st.error("❌ Please define at least peak hours or off hours!")
                    return
                
                config = {
                    'start_date': datetime.combine(start_date, time.min),
                    'end_date': datetime.combine(end_date, time.max),
                    'interval_minutes': interval_minutes,
                    'peak_hours': peak_hours,
                    'off_hours': off_hours,
                    'base_value': base_value,
                    'noise_level': noise_level,
                    'daily_seasonality': daily_seasonality,
                    'weekly_seasonality': weekly_seasonality,
                    'monthly_seasonality': monthly_seasonality,
                    'seasonality_strength': seasonality_strength,
                    'trends': trends,
                    'anomalies': anomalies
                }
                with st.spinner("🔄 Generating sophisticated synthetic time-series data..."):
                    df, total_anomaly_instances, total_anomalous_points = generate_synthetic_data(config)
                    st.session_state.generated_data = df
                    st.session_state.modified_data = df.copy()
                    st.session_state.total_anomaly_instances = total_anomaly_instances
                    st.session_state.total_anomalous_points = total_anomalous_points
        
        # Display generated data
        if st.session_state.generated_data is not None:
            df = st.session_state.generated_data
            
            st.header("📊 Generated Data Visualization")
            with st.expander("📈 Generated Data Visualization", expanded=False):
                st.markdown("<h3 style='color:#0072c6;'>📈 Generated Data Visualization</h3>", unsafe_allow_html=True)

                #st.subheader("📈 Generated Data Visualization")
    
                # Calculate total injected anomalies
                generated_anomalies = extract_generated_anomalies(df)  
                # total_anomalies = sum(info['count'] for info in generated_anomalies.values())
                total_anomalies = st.session_state.total_anomaly_instances
                

                # Display the enhanced metrics with detailed breakdown
                display_generated_anomaly_details(df, generated_anomalies, total_anomalies)  # TODO : This needs to be modified 

                # Interactive plot with click events
                fig = studio.create_interactive_plot(df, "GENERATE: Generated Synthetic Data with Ground Truth Anomalies")  # TODO : This needs to be modified 
            
                # Display the plot with click handling
                clicked_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="source_plot")
            
            # Point-by-point editing section
            st.header(" ✍️  Data Point Editor")
            
            col1, col2 = st.columns([1, 1])
            selected_indices = []
            
            with col1:
                #st.subheader("🎯 Edit Individual Points")
                with st.expander("🎯 Select Individual Points", expanded=False):
                    st.markdown("<h3 style='color:#0072c6;'>🎯 Select Individual Points</h3>", unsafe_allow_html=True)
                
                    # Point selection method
                    edit_method = st.radio(
                        "Selection Method:",
                        ["By Index", "By Timestamp", "Search Value Range"], horizontal=True,
                        help="Choose how to select points for editing"
                    )
                
                    if edit_method == "By Index":
                        index_input = st.text_input(
                            "Point Index (or range like 100-105):", 
                            help="Enter single index (e.g., 150) or range (e.g., 100-105)"
                        )
                        if index_input:
                            try:
                                if '-' in index_input:
                                    start_idx, end_idx = map(int, index_input.split('-'))
                                    selected_indices = list(range(max(0, start_idx), min(len(df), end_idx + 1)))
                                else:
                                    idx = int(index_input)
                                    if 0 <= idx < len(df):
                                        selected_indices = [idx]
                            except ValueError:
                                st.error("Invalid index format")
                
                    elif edit_method == "By Timestamp":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            edit_date = st.date_input("Select Date", df['timestamp'].dt.date.iloc[0])
                        with col_b:
                            edit_time = st.time_input("Select Time", time(12, 0))
                    
                        target_datetime = datetime.combine(edit_date, edit_time)
                        # Find closest timestamp
                        time_diffs = np.abs(df['timestamp'] - target_datetime)
                        closest_idx = time_diffs.idxmin()

                        selected_indices = [closest_idx]
                    
                        st.info(f"Closest point: Index {closest_idx}, Time: {df.loc[closest_idx, 'timestamp']}")
                
                    elif edit_method == "Search Value Range":
                        min_val = st.number_input("Min Value", value=float(df['value'].min()))
                        max_val = st.number_input("Max Value", value=float(df['value'].max()))
                    
                        mask = (df['value'] >= min_val) & (df['value'] <= max_val)
                        selected_indices = df[mask].index.tolist()
                    
                        if len(selected_indices) > 50:
                            st.warning(f"Found {len(selected_indices)} points. Showing first 50 for editing.")
                            selected_indices = selected_indices[:50]

            with col2:
                if selected_indices:
                    #st.subheader("🔧 Edit Selected Points")
                    with st.expander("🎯 Edit Selected Points", expanded=False):
                        st.markdown("<h3 style='color:#0072c6;'>🎯 Edit Selected Points</h3>", unsafe_allow_html=True)
                    
                        # Show selected points
                        selected_df = df.loc[selected_indices, ['timestamp', 'value', 'is_anomaly', 'anomaly_type']].copy()
                        st.dataframe(selected_df, use_container_width=True)
                    
                        # Bulk operations
                        st.subheader("⚡ Bulk Operations")
                    
                        col_a, col_b = st.columns(2)
                    
                        with col_a:
                            if st.button("🔴 Mark as Anomaly", use_container_width=True):
                                for idx in selected_indices:
                                    st.session_state.modified_data.at[idx, 'is_anomaly'] = True
                                    st.session_state.modified_data.at[idx, 'anomaly_type'] = 'manual'
                                st.success(f"✅ Marked {len(selected_indices)} points as anomalies")
                                st.rerun()
                    
                        with col_b:
                            if st.button("🟢 Mark as Normal", use_container_width=True):
                                for idx in selected_indices:
                                    st.session_state.modified_data.at[idx, 'is_anomaly'] = False
                                    st.session_state.modified_data.at[idx, 'anomaly_type'] = 'normal'
                                st.success(f"✅ Marked {len(selected_indices)} points as normal")
                                st.rerun()
                    
                        # Individual point editing
                        if len(selected_indices) == 1:
                            idx = selected_indices[0]
                            st.subheader(f"📝 Edit Point {idx}")
                        
                            current_is_anomaly = st.session_state.modified_data.at[idx, 'is_anomaly']
                            current_type = st.session_state.modified_data.at[idx, 'anomaly_type']
                            current_value = st.session_state.modified_data.at[idx, 'value']
                        
                            new_is_anomaly = st.checkbox(label = "check_anomaly", key="Is Anomaly", value=current_is_anomaly)
                        
                            if new_is_anomaly:
                                new_type = st.selectbox(
                                    "Anomaly Type",
                                    ["contextual", "pattern", "point", "manual"],
                                    index=["contextual", "pattern", "point", "manual"].index(current_type) if current_type in ["contextual", "pattern", "point", "manual"] else 3
                                )
                            else:
                                new_type = "normal"
                        
                            new_value = st.number_input("Value", value=float(current_value), step=0.1)
                        
                            if st.button("💾 Apply Changes", use_container_width=True):
                                st.session_state.modified_data.at[idx, 'is_anomaly'] = new_is_anomaly
                                st.session_state.modified_data.at[idx, 'anomaly_type'] = new_type
                                st.session_state.modified_data.at[idx, 'value'] = new_value
                                st.success("✅ Point updated successfully!")
                                st.rerun()
            
                # Show current modifications
                if st.session_state.modified_data is not None:
                    modified_df = st.session_state.modified_data
                    original_anomalies = st.session_state.generated_data['is_anomaly'].sum()
                    current_anomalies = modified_df['is_anomaly'].sum()
                
                    if current_anomalies != original_anomalies:
                        st.info(f"📝 **Modifications Applied**: Original anomalies: {original_anomalies}, Current: {current_anomalies}")
            
            # Export options
            st.header("💾 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export with fixed timestamp formatting
                if st.button("📄 Download CSV", type="primary", use_container_width=True):
                    try:
                        export_df = st.session_state.modified_data if st.session_state.modified_data is not None else df
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Save CSV File",
                            data=csv_data,
                            file_name=f"synthetic_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Export error: {e}")
            
            with col2:
                # JSON export with proper timestamp handling
                if st.button("📋 Download JSON", use_container_width=True):
                    try:
                        export_df = st.session_state.modified_data if st.session_state.modified_data is not None else df
                        json_df = export_df.copy()
                        # Fix the timestamp serialization issue
                        json_df['timestamp'] = json_df['timestamp'].astype(str)
                        json_str = json_df.to_json(orient='records', indent=2)
                        
                        st.download_button(
                            label="💾 Save JSON File",
                            data=json_str,
                            file_name=f"synthetic_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"JSON export error: {e}")
            
            with col3:
                # Configuration export
                if st.button("⚙️ Download Config", use_container_width=True):
                    config_export = {
                        'generation_timestamp': datetime.now().isoformat(),
                        'generation_config': {
                            'start_date': start_date.isoformat(),
                            'end_date': end_date.isoformat(),
                            'interval_minutes': interval_minutes,
                            'peak_hours': peak_hours,
                            'off_hours': off_hours,
                            'base_value': base_value,
                            'noise_level': noise_level,
                            'seasonality_strength': seasonality_strength,
                            'trends': trends,
                            'anomalies': anomalies
                        },
                        'data_statistics': {
                            'total_points': len(df),
                            'original_anomalies': st.session_state.generated_data['is_anomaly'].sum(),
                            'current_anomalies': st.session_state.modified_data['is_anomaly'].sum() if st.session_state.modified_data is not None else df['is_anomaly'].sum(),
                            'anomaly_rate': (df['is_anomaly'].mean() * 100),
                            'value_stats': {
                                'mean': df['value'].mean(),
                                'std': df['value'].std(),
                                'min': df['value'].min(),
                                'max': df['value'].max()
                            }
                        }
                    }
                    
                    print(f" config_export is : {config_export}")
                    # Pre-process the dictionary
                    processed_config = convert_numpy_types(config_export)
                    st.download_button(
                        label="💾 Save Config File",
                        data=json.dumps(processed_config, indent=2),
                        file_name=f"generation_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with tab2:
        st.header("🕵️ ANALYZE: Anomaly Detection Console")
        
        # Data input section
        #st.subheader("📁 Input Data")
        with st.expander("🗂️ Data Sourcing & Import", expanded=False):
            st.markdown("<h3 style='color:#0072c6;'>🗂️ Data Sourcing & Import </h3>", unsafe_allow_html=True)

        # with st.expander("📊 Data Profiler & Summary ", expanded=True):
        
            col1, col2 = st.columns(2)
        
            with col1:
                # Use generated data
                if st.session_state.modified_data is not None:
                    if st.button("📊 Use Generated Data", type="primary", use_container_width=True):
                        st.session_state.uploaded_data = st.session_state.modified_data
                        st.success("✅ Using generated synthetic data for analysis")
                        st.rerun()
        
            with col2:
                # Upload external data
                uploaded_file = st.file_uploader(
                    "📂 Upload CSV Data", 
                    type=['csv'],
                    help="Upload time-series data with columns: timestamp, value, is_anomaly (optional)"
                )
            
                if uploaded_file is not None:
                    try:
                        uploaded_df = pd.read_csv(uploaded_file)
                    
                        # Data validation and preprocessing
                        required_cols = ['timestamp', 'value']
                        missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
                    
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {missing_cols}")
                        else:
                            # Parse timestamp
                            uploaded_df['timestamp'] = pd.to_datetime(uploaded_df['timestamp'])
                        
                            # Add is_anomaly if not present
                            if 'is_anomaly' not in uploaded_df.columns:
                                uploaded_df['is_anomaly'] = False
                                st.warning("⚠️ No 'is_anomaly' column found. Added default False values.")
                        
                            # Add time features if not present
                            if 'hour' not in uploaded_df.columns:
                                uploaded_df['hour'] = uploaded_df['timestamp'].dt.hour
                            if 'day_of_week' not in uploaded_df.columns:
                                uploaded_df['day_of_week'] = uploaded_df['timestamp'].dt.dayofweek
                        
                            # Add missing cyclical features
                            if 'hour_sin' not in uploaded_df.columns:
                                uploaded_df['hour_sin'] = np.sin(2 * np.pi * uploaded_df['hour'] / 24)
                            if 'hour_cos' not in uploaded_df.columns:
                                uploaded_df['hour_cos'] = np.cos(2 * np.pi * uploaded_df['hour'] / 24)
                            if 'day_sin' not in uploaded_df.columns:
                                uploaded_df['day_sin'] = np.sin(2 * np.pi * uploaded_df['day_of_week'] / 7)
                            if 'day_cos' not in uploaded_df.columns:
                                uploaded_df['day_cos'] = np.cos(2 * np.pi * uploaded_df['day_of_week'] / 7)
                        
                            # Add boolean features if missing
                            for col in ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']:
                                if col not in uploaded_df.columns:
                                    uploaded_df[col] = False
                        
                            st.session_state.uploaded_data = uploaded_df
                            st.success(f"✅ Successfully loaded {len(uploaded_df):,} data points")
                        
                            # Show data preview
                            st.subheader("📋 Data Preview")
                            st.dataframe(uploaded_df.head(), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading file: {e}")
        
        if st.session_state.uploaded_data is not None:
            analysis_df = st.session_state.uploaded_data
            
            # st.subheader(" 📊 Data Profiler & Summary ")
            with st.expander(" 📊 Data Profiler & Summary ", expanded=False):
                st.markdown("<h3 style='color:#0072c6;'> 📊 Data Profiler & Summary </h3>", unsafe_allow_html=True)
                df = st.session_state.uploaded_data
           
                # --- Summary Metrics ---
                start_time = df["timestamp"].min()
                end_time = df["timestamp"].max()
                n_rows = len(df)

                # Frequency detection
                time_deltas = df["timestamp"].diff().dropna()
                most_common_interval = time_deltas.mode().iloc[0]
                interval_str = pd.to_timedelta(most_common_interval).components
                frequency_label = f"{interval_str.hours} hour(s) {interval_str.minutes} min(s)"

                # Value stats
                max_val = df["value"].max()
                min_val = df["value"].min()
                avg_val = df["value"].mean()

                st.subheader("🔍 Data Insights ")
                st.markdown("### 📌 Summary")
                summary_data = {
                    "Number of Rows": n_rows,
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Sampling Frequency": frequency_label,
                    "Max Value": round(max_val, 3),
                    "Min Value": round(min_val, 3),
                    "Average Value": round(avg_val, 3)
                }

                st.dataframe(pd.DataFrame(summary_data.items(), columns=["Property", "Value"]), use_container_width=True)

                # --- Graph Plot ---
                st.markdown("### 📈 Time Series Plot")
                st.line_chart(df.set_index("timestamp")["value"])

                # --- Subset Selection ---
                st.markdown("### ✂️ Data Trimming ")

                # Select by index or time
                view_type = st.radio("Subset by:", ["Top N Rows", "Time Range", "Index Range"], horizontal=True)

                print(f" DEBUG 1")
                if view_type == "Top N Rows":
                    top_n = st.slider("Select number of rows", 10, 200, 50)
                    st.dataframe(df.head(top_n), use_container_width=True)

                elif view_type == "Time Range":
                    start_time = df['timestamp'].min().to_pydatetime()  # This is a pd.Timestamp
                    end_time = df['timestamp'].max().to_pydatetime()    # Also a pd.Timestamp

                    default_start = start_time
                    default_end = start_time + timedelta(hours=200)

                    start, end = st.slider("Select time range", min_value=start_time, max_value=end_time,
                                           value=(default_start, default_end))
                    filtered = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
                    st.dataframe(filtered, use_container_width=True)

                elif view_type == "Index Range":
                    i_start, i_end = st.slider("Select index range", 0, len(df)-1, (0, 100))
                    st.dataframe(df.iloc[i_start:i_end+1], use_container_width=True)

                print(f" DEBUG 2 ")
                st.markdown("### 📈 Time Series Metadata ")

                props = analyze_data_properties(df,
                                                domain_knowledge_available=True,
                                                computational_budget='HIGH',
                                                real_time_requirement=False,
                                                interpretability_needed=True
                                                )
            
                print(f" DEBUG 3 ")
                # Get recommendations
                recommendations, data_summary = select_recommended_methods( props, include_rules=True, include_ml=True, max_methods=5)

                st.dataframe(pd.DataFrame(props.items(), columns=["Property", "Value"]), use_container_width=True)
                print(f" DEBUG 4 ")
                st.markdown("---")
                # UI: Select a property from list
                selected_property = st.selectbox("Click a property to learn more:", list(props.keys()))
                st.text(f"ℹ️ Details about **{selected_property}**")
                st.info(get_tooltip(selected_property))


        # Analysis section

        if st.session_state.uploaded_data is not None:
            analysis_df = st.session_state.uploaded_data
            
            # st.subheader(" ⚙️ Detection Algorithm Configurator ")
            with st.expander(" ⚙️ Detection Algorithm Configurator ", expanded=False):
                st.markdown("<h3 style='color:#0072c6;'> ⚙️ Detection Algorithm Configurator </h3>", unsafe_allow_html=True)
            
                col1, col2 = st.columns(2)
                df = st.session_state.uploaded_data
                #props = analyze_data_properties(df)
                props = analyze_data_properties(df,
                                                domain_knowledge_available=True,
                                                computational_budget='HIGH',
                                                real_time_requirement=False,
                                                interpretability_needed=True
                                                )
            
                # Get recommendations
                recommendations, data_summary = select_recommended_methods( props, include_rules=True, include_ml=True, max_methods=5)

                print("=== DATA SUMMARY FOR UI ===")
                for category, info in data_summary.items():
                    print(f"\n{category.upper()}:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")


                recommended_rules = []
                print("\n=== TOP RECOMMENDATIONS ===")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['method']} ({rec['type']})")
                    print(f"   Priority: {rec['priority']}, Score: {rec['final_score']:.1f}")
                    print(f"   Computational Cost: {rec['computational_cost']}")
                    print(f"   Real-time: {rec['real_time_capable']}, Interpretable: {rec['interpretable']}")
                    print()
                    recommended_rules.append(rec['method'])

                with col1:
                    st.markdown("**🔧 Rule-Based Methods**")
                    # rule_methods = studio.get_all_rule_methods(props)
                    rule_methods = RuleBasedAnomaly.get_all_rule_methods(props)
                
                    rule_categories = {}
                    selected_rules = {}

                    for rule_name, rule_info in rule_methods.items():
                        category = rule_info['category']
                        if category not in rule_categories:
                            rule_categories[category] = []
                        rule_categories[category].append((rule_name, rule_info))

                    for category, rules in rule_categories.items():
                        category_emoji = RuleBasedAnomaly.get_rule_category_emoji(category)
                        with st.expander(f"{category_emoji}{category.title()} Rules", expanded=True):
                            for rule_name, rule_info in rules:
                                label = rule_name.replace('_', ' ').title()
                                if rule_name in recommended_rules:
                                    label += " ⭐ (Recommended)"
                                else:
                                    print(f" rule_name{rule_name} not in recommended_rules : {recommended_rules}")

                                enabled = st.checkbox( label, value=rule_info['enabled'], key=f"rule_{rule_name}", help=studio.get_rule_helptext(rule_name))
                                if enabled:
                                    selected_rules[rule_name] = {'enabled': True}
                            rule_methods[rule_name]['enabled'] = enabled

                    rule_config = {}
                    for rule_name in selected_rules.keys():
                        with st.expander(f"⚙️ {rule_name.replace('_', ' ').title()} Settings"):
                            rule_config[rule_name] = studio.get_rule_config_ui(rule_name)
                            print(f"rule[{rule_name}] config is : {rule_config[rule_name]}")

                    print(f"selected_rules : {selected_rules}")

                with col2:
                    st.markdown("**🤖 ML-Based Methods**")

                    # props = analyze_data_properties(st.session_state.uploaded_data)

                    # all_ml_models = studio.get_all_ml_models(props)
                    all_ml_models = MLBasedAnomaly.get_all_ml_models(props)
                    selected_models = {}

                    # Group by categories
                    ml_categories = {}
                    for model_name, model_info in all_ml_models.items():
                        category = model_info['category']
                        if category not in ml_categories:
                            ml_categories[category] = []
                        ml_categories[category].append((model_name, model_info))

                    for category, models in ml_categories.items():
                        ml_category_emoji = MLBasedAnomaly.get_model_category_emoji(category)
                        with st.expander(f"{ml_category_emoji}{category.replace('_', ' ').title()} Models", expanded=True): 
                            for model_name, model_info in models:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    # how recommendation status
                                    label = model_name.replace('_', ' ').title()
                    
                                    if model_name in recommended_rules:
                                        label = f"{label} ⭐ (Recommended)"
                                    enabled = st.checkbox( label, value=model_info['enabled'], key=f"ml_{model_name}")


                                with col2:
                                    if enabled:
                                        weight = st.number_input(
                                            "Weight", 0.1, 3.0, model_info['weight'], 0.1,
                                            key=f"weight_{model_name}"
                                        )
                                        # Store with weight
                                        selected_models[model_name] = {'enabled': True, 'weight': weight, 'category': model_info['category'] }
                
                    ml_config = {}

                    # Usage in UI:
                    for model_name in selected_models.keys():
                        with st.expander(f"⚙️ {model_name.replace('_', ' ').title()} Settings"):
                            ml_config[model_name] = studio.get_model_config_ui(model_name)

                    # Add to DEST tab after model selection
                    st.subheader("🤝 Ensemble & Threshold Configuration")

                    col1, col2 = st.columns(2)
                    with col1:
                        use_ensemble = st.checkbox("Use Ensemble Scoring", value=True if len(selected_models) > 1 else False)
    
                        if use_ensemble:
                            ensemble_method = st.selectbox(
                                "Ensemble Method",
                                ['weighted_average', 'top_n', 'voting', 'median'],
                                help="Method to combine multiple model predictions"
                            )
                            if ensemble_method == 'top_n':
                                top_n_models = st.slider("Top N Models", 2, min(5, len(selected_models)), 3)

                    with col2:
                        use_adaptive_threshold = st.checkbox("Use Adaptive Threshold", value=True)
    
                        if use_adaptive_threshold:
                            threshold_method = st.selectbox(
                                "Threshold Method",
                                ['percentile', 'iqr', 'mad', 'std_based'],
                                help="Method to automatically determine anomaly threshold"
                            )
        
                            if threshold_method == 'percentile':
                                percentile_value = st.slider("Percentile", 90, 99, 95)
                            elif threshold_method == 'std_based':
                                std_multiplier = st.slider("Std Multiplier", 1.5, 4.0, 2.5)

            
            # Run analysis
            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                with st.spinner("🔄 Running anomaly detection algorithms..."):
                    
                    # Apply rule-based methods
                    if any(rule_config.values()):
                        print(f" selected rules for RULE-BASED are : {selected_rules}")
                        print(f" rule_config is : {rule_config}")
                        #st.session_state.rule_results = studio.apply_rule_based_detection(analysis_df, rule_methods, rule_config)
                        st.session_state.rule_results = RuleBasedAnomaly.apply_rule_based_detection(analysis_df, selected_rules, rule_config)
                    
                    # Apply ML-based methods
                    if any(ml_config.values()):
                        print(f" selected models for ML are : {selected_models}")
                        st.session_state.ml_results = MLBasedAnomaly.apply_ml_based_detection(analysis_df, selected_models, ml_config)

                st.write("Debug: ML Results content:")
                for method, anomalies in st.session_state.ml_results.items():
                    st.write(f"{method}: {anomalies.sum()} anomalies out of {len(anomalies)} total points")
                    st.write(f"Sample values: {anomalies[:10]}")
 
                    print(f"Method: {method}")
                    print(f"Any anomalies? {anomalies.any()}")
                    print(f"Count of True: {anomalies.sum()}")
                    print(f"Count of False: {(~anomalies).sum()}")

                    if anomalies.any():
                        anomaly_df = analysis_df[anomalies]
                        print(f"Filtered anomaly_df shape: {anomaly_df.shape}")
                    else:
                        print("No anomalies detected - skipping plot")

                st.success("✅ Anomaly detection completed!")
                st.rerun()
            
            # Display results
            if st.session_state.rule_results or st.session_state.ml_results:
                st.subheader(" ✔️ Anomaly Detection Report ")
                
                # Create visualization with all results
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Rule-Based Detection Results', 'ML-Based Detection Results'],
                    vertical_spacing=0.25,
                    row_heights=[0.5, 0.5] 
                )
                
                # Also add margins to the overall figure
                fig.update_layout(
                    height=800,  # Increase total height
                    margin=dict(t=60, b=60, l=60, r=60)  # Add margins around entire plot
                )

                # Original data
                fig.add_trace(
                    go.Scatter(
                        x=analysis_df['timestamp'],
                        y=analysis_df['value'],
                        mode='lines',
                        name='Original Data',
                        line=dict(color='blue', width=1),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=analysis_df['timestamp'],
                        y=analysis_df['value'],
                        mode='lines',
                        name='Original Data',
                        line=dict(color='blue', width=1),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Ground truth anomalies (if available)
                if 'is_anomaly' in analysis_df.columns and analysis_df['is_anomaly'].any():
                    analysis_df = ensure_boolean_anomaly_column(analysis_df, 'is_anomaly')
                    truth_df = analysis_df[analysis_df['is_anomaly']]
                    for row in [1, 2]:
                        fig.add_trace(
                            go.Scatter(
                                x=truth_df['timestamp'],
                                y=truth_df['value'],
                                mode='markers',
                                name='Ground Truth',
                                marker=dict(size=8, color='black', symbol='x'),
                                showlegend=(row==1)
                            ),
                            row=row, col=1
                        )
                
                # Rule-based results
                colors_rule = {'zscore': 'red', 'iqr': 'orange', 'moving_avg': 'purple'}
                for method, anomalies in st.session_state.rule_results.items():
                    if anomalies.any():
                        anomaly_df = analysis_df[anomalies]
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_df['timestamp'],
                                y=anomaly_df['value'],
                                mode='markers',
                                name=f'Rule: {method}',
                                marker=dict(size=6, color=colors_rule.get(method, 'red')),
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                
                # ML-based results  
                colors_ml = {'isolation_forest': 'green', 'one_class_svm': 'cyan', 'dbscan': 'magenta'}
                for method, anomalies in st.session_state.ml_results.items():
                    print(f"method is : {method}")
                    if anomalies.any():
                        anomaly_df = analysis_df[anomalies]
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_df['timestamp'],
                                y=anomaly_df['value'],
                                mode='markers',
                                name=f'ML: {method}',
                                marker=dict(size=6, color=colors_ml.get(method, 'green')),
                                showlegend=True
                            ),
                            row=2, col=1
                        )
                
                fig.update_layout(
                    height=800,
                    title="Comparision: Anomaly Detection Results Comparison",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🏆 Model & Rule Performance Comparison")
        
        if (st.session_state.uploaded_data is not None and 
            'is_anomaly' in st.session_state.uploaded_data.columns and
            (st.session_state.rule_results or st.session_state.ml_results)):
            
            analysis_df = st.session_state.uploaded_data
            ground_truth = analysis_df['is_anomaly'].values
            
            st.subheader("📊 Performance Metrics summary")
            
            # Calculate metrics for all methods
            all_results = {}
            all_results.update({f"Rule: {k}": v for k, v in st.session_state.rule_results.items()})
            all_results.update({f"ML: {k}": v for k, v in st.session_state.ml_results.items()})
            
            metrics_data = []
            
            for method_name, predictions in all_results.items():
                if predictions is not None and len(predictions) > 0:
                    #metrics = studio.calculate_metrics(ground_truth, predictions.values)
                    metrics = calculate_metrics(ground_truth, predictions)
                    if metrics:
                        metrics['method'] = method_name
                        metrics['type'] = 'Rule-Based' if method_name.startswith('Rule:') else 'ML-Based'
                        metrics_data.append(metrics)
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Display metrics table
                display_metrics = metrics_df[['method', 'type', 'accuracy', 'precision', 'recall', 'f1_score']].copy()
                display_metrics['accuracy'] = display_metrics['accuracy'].round(3)
                display_metrics['precision'] = display_metrics['precision'].round(3) 
                display_metrics['recall'] = display_metrics['recall'].round(3)
                display_metrics['f1_score'] = display_metrics['f1_score'].round(3)
                
                # Create an expander to hold the help text
                with st.expander(" 💡 Metric Definitions "):
                    st.markdown("**Accuracy:** The percentage of total predictions that the model got right ")
                    with st.expander("Example"):
                        st.write("""
                            Imagine you have 1000 data points, with 950 Normal and 50 actual anomalies.
                            The model predicted 930 normal points correctly and 30 anomalies correctly.
                            So, Accuracy = (930 + 30) / 1000 = 960/1000 = 96%.
                        """)

                    st.markdown("**Precision**: Of all the points the model labeled as anomalies, how many were actually anomalies? A high value means a low rate of false alarms. ")
                    with st.expander("Example"):
                        st.write("""
                            Imagine you have 1000 data points, with 950 Normal and 50 actual anomalies.
                            The model predicted 40 data points as anomalies out of that only 30 data points are actually anomalies. 
                            So, Precision = 30 correct anomalies / 40 predicted anomalies = 75%
                        """)

                    st.markdown("**Recall**: Of all the real anomalies, how many did the model successfully find ? A high value means the model misses very few anomalies")
                    with st.expander("Example"):
                        st.write("""
                            Imagine you have 1000 data points, with 950 Normal and 50 actual anomalies.
                            The model predicted 40 data points as anomalies out of that only 30 data points are actually anomalies. 
                            So, Recall = 30 correct anomalies / 50 Actual anomalies = 60%
                        """)

                    st.markdown("**F1-Score**: The harmonic mean of precision and recall. It's a balanced metric that gives a more comprehensive view of the model's performance, especially for imbalanced data (e.g., few anomalies).")
                    with st.expander("Example"):
                        st.write("""
                            Imagine you have 1000 data points, with 950 Normal and 50 actual anomalies.
                            The model predicted 40 data points as anomalies out of that only 30 data points are actually anomalies. 
                            So, precision is 30/40 = 75% and Recall = 30/50 = 60%, then F1-Score is calculated as : 2*(precision*recall / (precision+recall)) = 2 (0.75*0.60/(0.75+0.60)) = 2 * (0.45/1.35) = 0.67 = 67%. High F1-Score means your model is doing well both at finding true anomalies and avoiding false alarms — it’s balanced and reliable.Low F1-Score means your model is struggling with either catching enough anomalies (low recall) or making too many false alarms (low precision), so overall performance is weak. 
                        """)

                st.write("### Anomaly Detection Performance Metrics")
                st.dataframe(display_metrics, use_container_width=True)
                
                # Metrics visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Performance comparison
                    fig_perf = go.Figure()
                    
                    x_pos = np.arange(len(metrics_df))
                    
                    fig_perf.add_trace(go.Bar(name='Precision', x=metrics_df['method'], y=metrics_df['precision']))
                    fig_perf.add_trace(go.Bar(name='Recall', x=metrics_df['method'], y=metrics_df['recall']))  
                    fig_perf.add_trace(go.Bar(name='F1-Score', x=metrics_df['method'], y=metrics_df['f1_score']))
                    
                    fig_perf.update_layout(
                        title=' 📊 Model Performance Comparison',
                        xaxis_title='Detection Method',
                        yaxis_title='Score',
                        barmode='group',
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                with col2:
                    # Confusion matrix heatmap for best performing method
                    best_method_idx = metrics_df['f1_score'].idxmax()
                    best_method = metrics_df.loc[best_method_idx]
                    
                    # Create confusion matrix
                    cm_data = [[best_method['true_negatives'], best_method['false_positives']],
                              [best_method['false_negatives'], best_method['true_positives']]]
                    
                    fig_cm = px.imshow(
                        cm_data,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Normal', 'Anomaly'],
                        y=['Normal', 'Anomaly'],
                        color_continuous_scale='Blues',
                        title=f'📌 Confusion Matrix : {best_method["method"]}'
                    )
                    
                    # Add text annotations
                    for i in range(2):
                        for j in range(2):
                            fig_cm.add_annotation(
                                x=j, y=i,
                                text=str(cm_data[i][j]),
                                showarrow=False,
                                font=dict(color="white" if cm_data[i][j] > max(max(cm_data))*0.5 else "black")
                            )
                    
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                # Detailed comparison section
                st.subheader("🔬 Detailed Analysis")
                
                # Method comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏆 Best Performing Methods:**")
                    
                    best_accuracy = metrics_df.loc[metrics_df['accuracy'].idxmax()]
                    best_precision = metrics_df.loc[metrics_df['precision'].idxmax()]
                    best_recall = metrics_df.loc[metrics_df['recall'].idxmax()]
                    best_f1 = metrics_df.loc[metrics_df['f1_score'].idxmax()]
                    
                    st.write(f"🎯 **Accuracy**: {best_accuracy['method']} ({best_accuracy['accuracy']:.3f})")
                    st.write(f"🎯 **Precision**: {best_precision['method']} ({best_precision['precision']:.3f})")  
                    st.write(f"🎯 **Recall**: {best_recall['method']} ({best_recall['recall']:.3f})")
                    st.write(f"🎯 **F1-Score**: {best_f1['method']} ({best_f1['f1_score']:.3f})")
                
                with col2:
                    st.markdown("**📊 Performance Benchmarks :**")
                    
                    rule_methods = metrics_df[metrics_df['type'] == 'Rule-Based']
                    ml_methods = metrics_df[metrics_df['type'] == 'ML-Based']
                    
                    if len(rule_methods) > 0:
                        rule_avg_f1 = rule_methods['f1_score'].mean()
                        st.write(f"🛠️ **Rule-Based Average F1 Score**: {rule_avg_f1:.3f}")
                    
                    if len(ml_methods) > 0:
                        ml_avg_f1 = ml_methods['f1_score'].mean()
                        st.write(f"🤖 **ML-Based Average F1 Score**: {ml_avg_f1:.3f}")
                    
                    if len(rule_methods) > 0 and len(ml_methods) > 0:
                        advantage = ml_avg_f1 - rule_avg_f1
                        if advantage > 0:
                            st.write(f"📈 **ML Performance Advantage** : +{advantage:.3f}")
                        else:
                            st.write(f"📉 **Rule Performance Advantage**: +{-advantage:.3f}")
                
                # Add detailed analysis section
                if st.checkbox("Show Detailed Analysis"):
                    selected_method = st.selectbox("Select a Model for Analysis", list(all_results.keys()), key="method_selector")
                    if selected_method in all_results:
                        # Add this to force recalculation on selection change
                        if 'last_selected_method' not in st.session_state:
                            st.session_state.last_selected_method = None

                        if st.session_state.last_selected_method != selected_method:
                            st.session_state.last_selected_method = selected_method
                            # Clear any cached detailed metrics
                            if 'detailed_metrics_cache' in st.session_state:
                                del st.session_state.detailed_metrics_cache

                        #detailed_metrics = studio.calculate_detailed_metrics(
                        detailed_metrics = calculate_detailed_metrics(
                            analysis_df, 
                            all_results[selected_method], 
                            selected_method
                    )
        
                    col1, col2, col3 = st.columns(3)
        
                    with col1:
                        st.subheader("By Anomaly Type")
                        for atype, metrics in detailed_metrics['by_anomaly_type'].items():
                            st.write(f"**{atype.title()}**: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, FP={metrics['false_positives']},FN={metrics['false_negatives']}")
        
                    with col2:
                        st.subheader("By Time Period")
                        for period, metrics in detailed_metrics['by_time_period'].items():
                            st.write(f"**{period.title()}**: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, FP={metrics['false_positives']},FN={metrics['false_negatives']}")
        
                    with col3:
                        st.subheader("Confusion Matrix Details")
                        om = detailed_metrics['overall']
                        st.write(f"True Positives: {om['true_positives']}")
                        st.write(f"False Positives: {om['false_positives']}")
                        st.write(f"False Negatives: {om['false_negatives']}")
                        st.write(f"True Negatives: {om['true_negatives']}")


                # Agreement analysis
                st.subheader("🤝 Method Agreement Analysis")
                
                
                if len(all_results) >= 2:
                    method_names = list(all_results.keys())
                    n_methods = len(method_names)
                    agreement_matrix = np.zeros((n_methods, n_methods))
                    agreement_counts = np.zeros((n_methods, n_methods))
                    total_samples = len(next(iter(all_results.values())))  # assuming all have same length
                    
                    for i, method1 in enumerate(method_names):
                        for j, method2 in enumerate(method_names):
                            if i != j:
                                pred1 = all_results[method1]
                                pred2 = all_results[method2]
                                agreement_matrix[i][j] = np.mean(pred1 == pred2)
                                agreement_counts[i][j] = int(np.sum(pred1 == pred2))
                            else:
                                agreement_matrix[i][j] = 1.0
                                agreement_counts[i][j] = total_samples
                    
                    fig_agreement = px.imshow(
                        agreement_matrix,
                        labels=dict(x="Method", y="Method", color="Agreement"),
                        x=method_names,
                        y=method_names,
                        color_continuous_scale='RdYlBu',
                        title='Method Agreement Matrix'
                    )
                    
                    # Add text annotations
                    for i in range(n_methods):
                        for j in range(n_methods):
                            fig_agreement.add_annotation(
                                x=j, y=i,
                                text=f"{agreement_matrix[i][j]:.2f}%<br>({int(agreement_counts[i][j])}/{total_samples})",
                                showarrow=False,
                                font=dict(size=10, color='white')
                            )
                    
                    fig_agreement.update_layout(height=700, width=700,xaxis=dict(tickfont=dict(size=12)), yaxis=dict(tickfont=dict(size=12)) )
                    st.plotly_chart(fig_agreement, use_container_width=True)
                
                # Export results
                st.subheader("📤 Export Evaluation Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export metrics
                    if st.button("📊 Export Metrics CSV", use_container_width=True):
                        csv_data = metrics_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Metrics",
                            data=csv_data,
                            file_name=f"anomaly_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    # Export detailed results
                    if st.button("📋 Export Detailed Results", use_container_width=True):
                        detailed_results = analysis_df.copy()
                        
                        # Add prediction columns
                        for method_name, predictions in all_results.items():
                            # detailed_results[f"pred_{method_name}"] = predictions.values
                            detailed_results[f"pred_{method_name}"] = predictions
                        
                        csv_data = detailed_results.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Detailed Results",
                            data=csv_data,
                            file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            else:
                st.warning("⚠️ No valid results to display. Please run anomaly detection first.")
        
        else:
            st.info("📋 Please complete the following steps:")
            st.write("1. Generate synthetic data in the GENERATE tab OR upload data in ANALYZE tab")
            st.write("2. Run anomaly detection in the ANALYZE tab")
            st.write("3. Results will appear here for comparison")

    with tab4:
        st.header("📚 Help & Anomlytics Guide")
    
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            ### **Step-by-Step Workflow**
            1. **📊 GENERATE Tab - Generate Data:**
               - Configure date range and time intervals
               - Set peak/off hours for your use case
               - Adjust seasonality and trend parameters
               - Configure anomaly types and rates
               - Click "Generate Synthetic Data"
               - Visualize the Generated Data w.r.t different categories of anomalies in that data
               - For Manual editing of any data sample, please refer the "Manual Editing" section.
               - Export the data or config for future usage
            2. **🔍 ANALYZE Tab - Run Anomaly Detection:**
               - Choose to use generated data or upload your own CSV
               - Select & Configure rule-based methods
               - Select & Configure ML models (configure ensemble & Threshold , for multiple ML Models usage)
               - Click "Run Anomaly Detection"
               - Detection Results : contains the comparision between anomaly detection by Rule-Based and ML-Based selected methods
            3. **📈 COMPARE Tab - Analyze Results:**
               - View performance metrics for all methods
               - Compare precision, recall, F1-scores
               - Analyze method agreements
               - View the performance of individual Rule-Methods and ML-Models w.r.t different anomaly categories and time
               - Export results for further analysis
           4. **✏️ Manual Editing (Optional):**
               - In GENERATE tab, use point-by-point editing
               - Select points by index, timestamp, or value range
               - Mark points as anomalies or normal
               - Apply bulk operations
            """)
    
        # NEW: Data Preprocessing Methods
        with st.expander("🔧 Data Preprocessing Methods"):
            st.markdown("""
            ### **Missing Data Handling**
            **Forward Fill**
            - Fills missing values with the last observed value
            - Example: [1, 2, NaN, 4] → [1, 2, 2, 4]
            - Best for: Stable trends, sensor data
        
            **Backward Fill**
            - Fills missing values with the next observed value
            - Example: [1, NaN, 3, 4] → [1, 3, 3, 4]
            - Best for: End-of-period adjustments
        
            **Linear Interpolation**
            - Estimates missing values using linear relationship between neighboring points
            - Example: [1, NaN, 5] → [1, 3, 5]
            - Best for: Smooth trends, temperature data
        
            **Seasonal Decomposition**
            - Uses seasonal patterns to estimate missing values
            - Example: If Mondays typically have 20% higher values, apply this pattern
            - Best for: Strong seasonal data (retail sales, web traffic)
        
            **ML-based Imputation**
            - Uses machine learning models (KNN, Random Forest) to predict missing values
            - Considers multiple features and complex patterns
            - Best for: Complex multivariate time series
        
            ### **Noise Reduction**
            **Moving Average**
            - Smooths data by averaging neighboring points
            - Example: 3-point MA of [1,5,2,8,3] → [2.67,5,4.33]
            - Best for: Removing high-frequency noise
        
            **Savitzky-Golay Filter**
            - Polynomial-based smoothing that preserves peaks
            - Better than moving average for maintaining shape
            - Best for: Scientific measurements, spectroscopy data
        
            **Kalman Filter**
            - Optimal estimator that predicts and corrects based on uncertainty
            - Adapts to changing noise levels
            - Best for: GPS tracking, sensor fusion
        
            **Wavelet Denoising**
            - Decomposes signal into frequency components, removes noise frequencies
            - Preserves important signal features
            - Best for: Audio signals, biomedical data
        
            ### **Outlier Handling**
            **Winsorization**
            - Caps extreme values at specified percentiles
            - Example: Cap values below 5th percentile and above 95th percentile
            - Best for: Financial data, survey responses
        
            **Transformation**
            - Applies mathematical transformation to reduce outlier impact
            - Example: Log transformation for right-skewed data
            - Best for: Revenue data, population metrics

            ### **Resampling Methods**
            **Time-Based Resampling**
            **Upsampling (Increasing Frequency)**
            - Increases data points by interpolating between existing values
            - Example: Hourly [10, 20, 30] → Half-hourly [10, 15, 20, 25, 30]
            - Best for: Creating smoother signals, matching higher frequency data

            **Downsampling (Decreasing Frequency)**
            - Reduces data points by aggregating over larger time windows
            - Example: Hourly [10, 12, 14, 16, 18, 20] → Daily [15] (mean of 6 hours)
            - Best for: Reducing computational load, focusing on trends

            **Forward Fill (ffill)**
            - Propagates last valid observation forward
            - Example: [10, NaN, NaN, 20] → [10, 10, 10, 20]
            - Best for: Step-wise processes, constant states between measurements

            **Backward Fill (bfill)**
            - Propagates next valid observation backward
            - Example: [10, NaN, NaN, 20] → [10, 20, 20, 20]
            - Best for: When future values better represent missing periods

            **Interpolation Methods**
            **Linear Interpolation**
            - Draws straight lines between known points
            - Example: [10, NaN, NaN, 20] at times [0, 1, 2, 3] → [10, 13.33, 16.67, 20]
            - Best for: Smooth transitions, gradually changing values

            **Polynomial Interpolation**
            - Fits polynomial curves through data points
            - Example: [1, NaN, 9] → Quadratic fit → [1, 4, 9] (x² pattern)
            - Best for: Curved relationships, non-linear trends

            **Spline Interpolation**
            - Uses piecewise polynomial curves for smooth interpolation
            - Example: Smooth curve through [1, 4, 2, 8] with natural boundaries
            - Best for: Complex curves, maintaining smoothness

            **Aggregation Methods**
            **Mean/Average**
            - Takes arithmetic mean of values in time window
            - Example: [10, 12, 14, 16] (4 hours) → [13] (daily average)
            - Best for: General trend analysis, normal distributions

            **Median**
            - Takes middle value of sorted data in window
            - Example: [10, 12, 100, 16] → [14] (robust to outlier 100)
            - Best for: Data with outliers, skewed distributions

            **Sum/Total**
            - Adds all values in time window
            - Example: [10, 20, 30, 40] (hourly sales) → [100] (daily total)
            - Best for: Cumulative metrics (sales, counts, volumes)

            **Min/Max**
            - Takes minimum or maximum value in window
            - Example: [18, 25, 22, 19] (hourly temp) → Min:[18], Max:[25]
            - Best for: Extreme value tracking, peak/valley detection

            **First/Last**
            - Takes first or last value in time window
            - Example: [100, 105, 98, 102] (stock prices) → First:[100], Last:[102]
            - Best for: Opening/closing values, state at specific times

            **Statistical Resampling**
            **OHLC (Open-High-Low-Close)**
            - Financial resampling showing key statistics per period
            - Example: Minute prices [100, 105, 98, 102, 104] → OHLC:[100, 105, 98, 104]
            - Best for: Financial data, capturing price action summary

            **Standard Deviation**
            - Measures variability within each time window
            - Example: [10, 12, 8, 14] → [2.58] (volatility measure)
            - Best for: Risk metrics, stability assessment

            **Percentiles**
            - Takes specific percentiles (25th, 75th, etc.) from window
            - Example: [1, 5, 10, 15, 20] → 75th percentile: [15]
            - Best for: Distribution analysis, outlier-resistant summaries

            **Advanced Resampling**
            **Seasonal Decomposition + Resample**
            - Separates trend/seasonal/residual, then resamples each component
            - Example: Decompose daily data → resample trend monthly, seasonality weekly
            - Best for: Multi-scale analysis, preserving seasonal patterns

            **Wavelet-Based Resampling**
            - Uses wavelet transforms for multi-resolution analysis
            - Example: Preserve both high-freq noise and low-freq trends at different scales
            - Best for: Signal processing, preserving multiple time scales

            **Model-Based Resampling**
            - Fits model (ARIMA, etc.) then generates values at new frequency
            - Example: Fit ARIMA to hourly data → predict half-hourly values
            - Best for: When interpolation needs to capture complex patterns

            **Practical Selection Guide**
            - For Missing Values: Forward/Backward Fill → Linear Interpolation → Spline
            - For Downsampling: Mean (trends) → Median (outliers) → OHLC (finance)
            - For Upsampling: Linear → Polynomial → Model-based (complexity increasing)
            - For Aggregation: Sum (totals) → Mean (averages) → Min/Max (extremes)

        
            ### **Normalization Methods**
            **Min-Max Scaling**
            - Scales values to [0,1] range: (x - min) / (max - min)
            - Example: [10,20,30] → [0, 0.5, 1]
            - Best for: Neural networks, when bounds are known
        
            **Z-Score Standardization**
            - Centers data around mean with unit variance: (x - μ) / σ
            - Example: [10,20,30] with μ=20, σ=10 → [-1, 0, 1]
            - Best for: Normally distributed data, statistical models
        
            **Robust Scaling**
            - Uses median and IQR instead of mean and std dev
            - Less sensitive to outliers
            - Best for: Data with outliers, non-normal distributions
            """)
    
        # NEW: Data Properties & Model Selection
        with st.expander("📊 Data Properties & Model Selection Guide"):
            st.markdown("""
            ### **Understanding Your Data Properties**
        
            **Temporal Properties**
            - **Sampling Rate**: Frequency of data collection (e.g., 1 sample/minute)
            - **Data Frequency**: Regular intervals vs irregular timestamps
            - **Time Range**: Total period covered affects seasonal analysis capabilities
        
            **Statistical Properties**
            - **Distribution Type**: Normal, log-normal, exponential, etc.
            - **Skewness/Kurtosis**: Measures asymmetry and tail heaviness
            - **High skewness** → consider transformations
            - **High kurtosis** → expect more outliers
        
            **Trend Properties**
            - **Trend Direction**: Increasing, decreasing, or stable over time
            - **Change Points**: Points where trend direction changes abruptly
            - Example: COVID-19 impact on retail sales in March 2020
        
            **Seasonality Properties**
            - **Seasonal Periods**: Daily (24h), Weekly (7d), Yearly (365d)
            - **Multiple Seasonality**: Data with multiple seasonal patterns
            - Example: Electricity demand (daily + weekly + yearly patterns)
        
            **Stationarity Properties**
            - **ADF Test**: p-value < 0.05 suggests stationarity
            - **KPSS Test**: p-value > 0.05 suggests stationarity
        
            ### **Property-Based Model Selection**
        
            **High Seasonality → Seasonal Models**
            - **When**: Strong recurring patterns detected
            - **Models**: SARIMA, Seasonal LSTM, Prophet
            - **Example**: Retail sales, energy consumption
        
            **High Trend → Trend-Based Models**
            - **When**: Persistent upward/downward movement
            - **Models**: Linear regression, ARIMA with differencing
            - **Example**: Population growth, technology adoption
        
            **High Volatility → Robust Models**
            - **When**: High variance, many outliers
            - **Models**: IQR-based rules, Isolation Forest
            - **Example**: Stock prices, network traffic
        
            **Stationary + Normal → Classical Statistical**
            - **When**: Stable mean/variance, normal distribution
            - **Models**: Z-score, control charts, ARIMA
            - **Example**: Manufacturing quality metrics
            """)
    
        # NEW: Ensemble Methods & Decision Making
        with st.expander("🎯 Ensemble Methods & Decision Making"):
            st.markdown("""
            ### **Weight Assignment Methods**
        
            **Performance-Based Weights**
            - Assigns higher weights to better-performing models
            - **Metrics**: Accuracy, F1-score, ROC-AUC
            - **Example**: Model A (90% accuracy) gets weight 0.6, Model B (70% accuracy) gets 0.4
        
            **Property-Based Weights**
            - Weights based on how well model fits data characteristics
            - **Example**: Seasonal model gets higher weight for seasonal data
        
            **Dynamic Weights**
            - Weights change over time based on recent performance
            - **Example**: Model weight decreases if recent predictions are poor
        
            **Equal Weights**
            - All models treated equally (democratic voting)
            - **Example**: 3 models each get weight 0.33
        
            ### **Weight Application Timing**
        
            **Pre-Aggregation Weighting**
            - Apply weights before combining scores
            - **Formula**: Final_Score = W1×S1 + W2×S2 + W3×S3
            - **Example**: 0.5×0.8 + 0.3×0.6 + 0.2×0.9 = 0.76
        
            **Post-Aggregation Weighting**
            - Combine scores first, then apply weight
            - **Formula**: Final_Score = Average(S1,S2,S3) × Weight
            - **Example**: (0.8+0.6+0.9)/3 × 0.8 = 0.61
        
            **Multi-Stage Weighting**
            - Hierarchical weighting (category level, then model level)
            - **Example**: Statistical models (0.4) × Individual Z-score weight (0.6) = 0.24
        
            ### **🤝 Ensemble & Aggregation**
        
            **Weighted Average**
            - **Formula**: Σ(Wi × Si) / ΣWi
            - **Example**: (0.5×0.8 + 0.3×0.6) / (0.5+0.3) = 0.725
            - **Best for**: Continuous scores, importance-based combination
        
            **Majority Voting**
            - Each model votes "anomaly" or "normal", majority wins
            - **Example**: 3 models vote [Anomaly, Normal, Anomaly] → Final: Anomaly
            - **Best for**: Binary decisions, democratic approach
        
            **Weighted Voting**
            - Weighted votes based on model importance
            - **Example**: Model votes [A,N,A] with weights [0.5,0.2,0.3] → 0.8 for Anomaly
            - **Best for**: Combining model importance with voting
        
            **Rank-Based Fusion**
            - Converts scores to ranks, then combines
            - **Example**: Scores [0.9,0.7,0.8] → Ranks [1,3,2] → Average rank = 2
            - **Best for**: When score scales differ across models
        
            **Meta-Learning**
            - Train ML model to learn optimal combination
            - **Example**: Neural network learns: Final = f(S1,S2,S3,context)
            - **Best for**: Complex patterns, large datasets
        
            ### **Final Threshold Generation**
        
            **Statistical Threshold**
            - **Mean ± K×StdDev**: Common choice K=2 or 3
            - **Percentile-based**: 95th percentile as threshold
            - **Example**: If scores have μ=0.3, σ=0.1, then threshold = 0.3 + 2×0.1 = 0.5
        
            **Historical Threshold**
            - Based on past anomaly rates
            - **Example**: If historically 5% are anomalies, set threshold at 95th percentile
            - **Best for**: Maintaining consistent anomaly rates
        
            **Dynamic Threshold**
            - Adapts based on recent performance and feedback
            - **Example**: Lower threshold if missing important anomalies
            - **Best for**: Changing environments, concept drift
        
            **Optimization-Based**
            - **ROC Optimization**: Maximize true positive rate - false positive rate
            - **F1 Optimization**: Maximize harmonic mean of precision and recall
            - **Cost-Sensitive**: Minimize business cost (false alarms vs missed anomalies)
            """)
    
        # NEW: Individual vs Collective Analysis
        with st.expander("⚖️ Individual vs Collective Analysis"):
            st.markdown("""
            ### **Individual Analysis Approach**
            - Each model makes independent decisions
            - **Pros**: Simple, interpretable, fast
            - **Cons**: May miss consensus patterns
            - **Example**: Z-score > 3 OR IQR outlier OR trend break detected
            - **Use when**: Need quick decisions, models are highly specialized
        
            ### **Collective Analysis Approach**
            - Models work together for final decision
            - **Pros**: More robust, reduces false positives
            - **Cons**: Complex, requires tuning
            - **Example**: Weighted combination of all model scores > ensemble threshold
            - **Use when**: High accuracy required, have diverse models
        
            ### **Hybrid Approach**
            - Combine both individual and collective insights
            - **Individual alerts**: For immediate action on high-confidence detections
            - **Collective consensus**: For final decision and explanation
            - **Best practice**: Use individual for real-time alerts, collective for analysis
            """)
    
        # Comprehensive glossary (existing)
        studio.show_glossary()
    
        # Enhanced troubleshooting section
        with st.expander("🛠️ Troubleshooting & Best Practices"):
            st.markdown("""
            ### **❌ Common Errors & Fixes**
            **🚨 "End date must be after start date" Error:**
            - Make sure your end date is later than your start date
            - Check that dates are in the correct format
        
            **🚨 "Please define at least peak hours or off hours" Error:**
            - Enter time ranges like "8-10,17-19" for peak hours
            - Or enter "0-5,22-23" for off hours
            - At least one must be defined
        
            **🚨 📉 Performance Degradation:**
            - Try adjusting anomaly rates (lower = more realistic)
            - Experiment with different intensity values
            - Consider using multiple detection methods
            - Check if your data properties match selected models
        
            **🚨 Memory Issues with Large Datasets:**
            - Reduce the date range
            - Increase data intervals (use 60min instead of 15min)
            - Consider sampling your data
        
            **🚨 Poor Ensemble Performance:**
            - Check individual model performances first
            - Adjust weight assignment method
            - Try different threshold generation approaches
            - Ensure models are diverse (different strengths)
        
            ### **Best Practices**
            **📊 Data Generation:**
            - Start with realistic anomaly rates (1-3%)
            - Use domain-appropriate peak/off hours
            - Adjust seasonality based on your use case
        
            **🔍 Detection Methods:**
            - Use multiple methods for comparison
            - Rule-based methods are good baselines
            - ML methods often perform better on complex patterns
            - Consider data properties when selecting methods
        
            **📈 Evaluation:**
            - Focus on F1-score for balanced performance
            - High precision = fewer false alarms
            - High recall = catches more real anomalies
            - Monitor performance over time for concept drift
        
            **🎯 Ensemble Configuration:**
            - Start with equal weights, then optimize
            - Use performance-based weights for stable environments
            - Use dynamic weights for changing conditions
            - Validate ensemble performance against individual models
            """)
    
        # Enhanced advanced usage section
        with st.expander("🎓 Advanced Usage"):
            st.markdown("""
            ### **Parameter Tuning Guide**
            **Seasonality Strengths:**
            - Daily: 10-30 for moderate variation, 50+ for extreme
            - Weekly: 5-20 for subtle patterns, 30+ for strong
            - Monthly: 5-15 typically sufficient
        
            **Trend Multipliers:**
            - Peak hours: 1.2-2.0 (20%-100% increase)
            - Off hours: 0.3-0.8 (30%-70% decrease)
            - Weekends: 0.6-1.2 depending on domain
        
            **ML Algorithm Selection:**
            - **Isolation Forest**: Good general purpose, handles multiple anomaly types
            - **One-Class SVM**: Better for well-defined normal regions
            - **DBSCAN**: Excellent for density-based anomalies
        
            **Rule-Based Tuning:**
            - Z-score threshold: 2.5-3.5 for normal distributions
            - Moving window: 5-20 depending on data frequency
            - IQR method: Works well for skewed distributions
        
            **Ensemble Tuning:**
            - **Weight Assignment**: Start with performance-based, adjust for stability
            - **Threshold Selection**: Use ROC optimization for balanced performance
            - **Score Combination**: Weighted average for continuous scores, voting for binary
        
            ### **🌐 Industry Use Cases**
            **📈 Business Analytics:**
            - Website traffic, sales data, user engagement
            - Peak hours: business hours, lunch time
            - Seasonality: daily patterns, weekend effects
            - **Recommended**: Seasonal models + trend analysis
        
            **🏭 Industrial IoT:**
            - Sensor readings, equipment performance
            - Context-aware: maintenance schedules, shift patterns
            - Pattern anomalies: degradation trends
            - **Recommended**: Statistical models + robust methods
        
            **💰 Financial Data:**
            - Transaction volumes, price movements
            - Market hours, trading sessions
            - Point anomalies: flash crashes, spikes
            - **Recommended**: Volatility-aware models + ensemble
        
            **🌐 Network Monitoring:**
            - Bandwidth usage, connection counts
            - Peak hours: business vs off-hours
            - Contextual: unusual traffic during quiet periods
            - **Recommended**: Multi-threshold rules + ML ensemble
        
            ### **🔁 Continuous Learning & Feedback**
            **🔄 Feedback Loop Automation:**
            - Collect expert validation regularly
            - Monitor performance metrics over time
            - Implement automatic retraining triggers
            - Update thresholds based on false positive/negative rates
        
            **⚠️ Model Drift Monitoring**
            - Track model performance degradation
            - Monitor data distribution changes
            - Implement concept drift detection
            - Set up automatic alerts for performance drops
            """)
    
        # Data format section (existing)
        with st.expander("📄 Data Format Requirements"):
            st.markdown("""
            ### **CSV Upload Format**
            **Required Columns:**
            - `timestamp`: Date/time in ISO format (YYYY-MM-DD HH:MM:SS)
            - `value`: Numeric values to analyze
        
            **Optional Columns:**
            - `is_anomaly`: Boolean (True/False) for ground truth labels
            - `anomaly_type`: String describing anomaly type
            - Time features (hour, day_of_week, etc.) - will be auto-generated
        
            **Example CSV Structure:**
            ```
            timestamp,value,is_anomaly,anomaly_type
            2024-01-01 00:00:00,95.2,False,normal
            2024-01-01 00:15:00,89.1,False,normal
            2024-01-01 00:30:00,156.7,True,point
            ```
        
            ### **Data Quality Requirements**
            **For Optimal Performance:**
            - **Completeness**: < 10% missing values recommended
            - **Consistency**: Regular time intervals preferred
            - **Range**: Check for physically impossible values
            - **Labels**: Ground truth labels improve evaluation accuracy
        
            ### **Export Formats**
            **CSV Export:** Full dataset with predictions from all methods
            **JSON Export:** Structured format for programmatic use
            **Config Export:** Complete generation parameters for reproducibility
        
            ### **File Size Limits**
            - Recommended: < 100,000 data points
            - Maximum: < 1,000,000 data points
            - For larger datasets, consider data sampling
            """)
    
        # Enhanced performance section
        with st.expander("⚡ Performance & Optimization"):
            st.markdown("""
            ### **Performance Tips**
            **Data Generation Speed:**
            - 15-minute intervals: ~96 points per day
            - 1-hour intervals: ~24 points per day
            - Longer intervals = faster generation
        
            **Detection Algorithm Speed:**
            - Rule-based methods: Very fast, real-time capable
            - Isolation Forest: Fast, scales well
            - One-Class SVM: Moderate speed
            - DBSCAN: Can be slow on large datasets
        
            **Ensemble Performance:**
            - **Pre-aggregation weighting**: Faster computation
            - **Equal weights**: Fastest ensemble method
            - **Meta-learning**: Slower but most accurate
            - **Majority voting**: Good balance of speed and accuracy
        
            **Memory Usage:**
            - Each data point: ~1KB with all features
            - 100K points: ~100MB memory usage
            - Monitor browser memory for large datasets
            - Ensemble methods use ~30% more memory
        
            ### **Scalability Considerations**
            **For Production Use:**
            - Implement streaming detection for real-time
            - Use batch processing for historical analysis
            - Consider model retraining frequency
            - Cache frequently used model weights
        
            **Cloud Deployment:**
            - Export configurations for reproducible pipelines
            - Use containerization for consistent environments
            - Implement proper logging and monitoring
            - Set up automated model performance tracking
        
            ### **Optimization Strategies**
            **Model Selection Optimization:**
            - Profile different models on your data
            - Use simpler models for real-time detection
            - Reserve complex ensembles for offline analysis
        
            **Threshold Optimization:**
            - Use historical data to optimize thresholds
            - Implement A/B testing for threshold changes
            - Monitor business metrics alongside technical metrics
            """)
    
        # NEW: Method Comparison Matrix
        with st.expander("📋 Method Comparison Matrix"):
            st.markdown("""
            ### **Detection Method Comparison**
        
            | Method | Speed | Accuracy | Interpretability | Data Requirements |
            |--------|-------|----------|------------------|-------------------|
            | **Z-Score** | ⚡⚡⚡ | ⭐⭐ | ⚡⚡⚡ | Normal distribution |
            | **IQR** | ⚡⚡⚡ | ⭐⭐⭐ | ⚡⚡⚡ | Any distribution |
            | **Isolation Forest** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ | Large datasets |
            | **One-Class SVM** | ⚡ | ⭐⭐⭐⭐ | ⭐ | Well-defined normal |
            | **DBSCAN** | ⚡ | ⭐⭐⭐ | ⭐⭐ | Density variations |
            | **Ensemble** | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Diverse methods |
        
            ### **Use Case Recommendations**
        
            **Real-time Monitoring** → Z-Score, IQR, Simple Rules
            **High Accuracy Needed** → Ensemble Methods, Multiple Models
            **Interpretability Critical** → Rule-based Methods, Statistical Tests
            **Complex Patterns** → ML Ensemble, Deep Learning
            **Limited Data** → Rule-based Methods, Simple Statistical
            **Streaming Data** → Lightweight Rules, Online Learning
            """)    


if __name__ == "__main__":
    main()
