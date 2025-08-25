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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from io import StringIO
import requests
from additional_anomaly_methods import apply_kmeans_detection, train_usad, usad_score, meta_aad_active_loop, get_torch_device
from feature_utils import _prepare_training_data, _adaptive_threshold, _detect_datetime_features

class MLBasedAnomaly:
    def __init__(self):
        all_models = {}

    @staticmethod
    def get_all_ml_models(props):
        """Get complete list of ML models including time-series models"""
        all_models = {
            # Isolation-based
            'isolation_forest': {'weight': 1.0, 'enabled': False, 'category': 'isolation'},
        
            # Distance-based
            'one_class_svm': {'weight': 1.0, 'enabled': False, 'category': 'boundary'},
            'elliptic_envelope': {'weight': 1.0, 'enabled': False, 'category': 'boundary'},

            # Density Based
            'local_outlier_factor': {'weight': 1.0, 'enabled': False, 'category': 'density'},
            'dbscan': {'weight': 1.0, 'enabled': False, 'category': 'density'},
        
            # Clustering-based
            'kmeans': {'weight': 1.0, 'enabled': False, 'category': 'clustering'},
        
            # Statistical
            'zscore': {'weight': 1.0, 'enabled': False, 'category': 'statistical_ml'},
            'mad': {'weight': 1.0, 'enabled': False, 'category': 'statistical_ml'},
            'ewma': {'weight': 1.0, 'enabled': False, 'category': 'statistical_ml'},
        
            # Time Series Models
            'prophet': {'weight': 1.0, 'enabled': False, 'category': 'time_series'},
            'arima': {'weight': 1.0, 'enabled': False, 'category': 'time_series'},
            'sarima': {'weight': 1.0, 'enabled': False, 'category': 'time_series'},
            #'auto_arima': {'weight': 1.0, 'enabled': False, 'category': 'time_series'},
            'exponential_smoothing': {'weight': 1.0, 'enabled': False, 'category': 'time_series'},
        
            # Deep learning
            'lstm': {'weight': 1.0, 'enabled': False, 'category': 'deep_learning'},
            'gru': {'weight': 1.0, 'enabled': False, 'category': 'deep_learning'},
            'usad': {'weight': 1.0, 'enabled': False, 'category': 'deep_learning'},
            'autoencoder': {'weight': 1.0, 'enabled': False, 'category': 'deep_learning'},
        }

        return all_models


    @staticmethod
    def get_model_category_emoji(model_category):
        if model_category == "isolation":
            return " 📦 "
        elif model_category == "boundary":
            return " 🚧 "
        elif model_category == "density":
            return " 🔬 "
        elif model_category == "clustering":
            return " 🔗 "
        elif model_category == "statistical_ml":
            return " 📈 "
        elif model_category == "time_series":
            return " ⏳ "
        elif model_category == "deep_learning":
            return " 🧠 "
        else:
            return ""


    @staticmethod
    def apply_prophet_detection(df, config):
        """Apply Prophet model for anomaly detection"""
        try:
            from prophet import Prophet

            prophet_config = config.get('prophet', {})
            print(f" apply_prophet_detection() : inputs - config is : {prophet_config}")
        
            # Prepare data for Prophet
            prophet_df = df[['timestamp', 'value']].copy()
            prophet_df.columns = ['ds', 'y']
        
            # Fit Prophet model
            model = Prophet(
                yearly_seasonality=prophet_config.get('yearly_seasonality', True),
                weekly_seasonality=prophet_config.get('weekly_seasonality', True),
                daily_seasonality=prophet_config.get('daily_seasonality', True),
                changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
                seasonality_mode = prophet_config.get('seasonality_mode', 'additive'),
                interval_width = prophet_config.get('interval_width', 0.8),

            )
            model.fit(prophet_df)
        
            # Make predictions
            forecast = model.predict(prophet_df)
        
            # Calculate residuals and detect anomalies
            residuals = np.abs(prophet_df['y'] - forecast['yhat'])

            # Detect anomalies based on residual threshold
            threshold_method =  prophet_config.get('threshold_method', 'percentile')
            threshold_value = prophet_config.get('threshold_value', 2.0)

            # Apply threshold to get binary anomaly flags
            if threshold_method == 'percentile':
                threshold = np.percentile(residuals, 95)
            elif result['threshold_method'] == 'std':
                threshold = np.mean(residuals) + threshold_value * np.std(residuals)
            else:
                threshold = 0.1
        
            anomalies = np.abs(residuals) > threshold

            df['prophet'] = anomalies
            df['prophet_anomaly_scores'] = residuals
        
            return anomalies, residuals
        
        except ImportError:
            print(f"Prophet not installed. Run: pip install prophet")
            return np.array([False] * len(df)), np.array([0] * len(df))

    @staticmethod
    def apply_arima_detection(df, config):
        """Apply ARIMA model for anomaly detection"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            arima_config = config.get('arima', {})
            print(f" apply_arima_detection() : inputs - config is : {arima_config}")
        
            # Fit ARIMA model
            order = (arima_config.get('p', 1), arima_config.get('d',1), arima_config.get('q', 1))
            model = ARIMA(df['value'], order=order)
            fitted_model = model.fit()
        
            # Get residuals
            residuals = fitted_model.resid
        
            # Detect anomalies based on residual threshold
            threshold_method =  arima_config.get('threshold_method', 'percentile')
            threshold_value = arima_config.get('threshold_value', 2.0)

            # Apply threshold to get binary anomaly flags
           
            if threshold_method == 'percentile':
                threshold = np.percentile(residuals, 95)
            elif threshold_method == 'std':
                threshold = np.mean(residuals) + threshold_value * np.std(residuals)
            else:
                threshold = 0.1
        
            anomalies = np.abs(residuals) > threshold
            df['arima'] = anomalies
            anomaly_scores = np.abs(residuals)
            df['arima_anomaly_scores'] = anomaly_scores
        
            return anomalies, anomaly_scores
        
        except ImportError:
            print(f"statsmodels not installed. Run: pip install statsmodels")
            return np.array([False] * len(df)), np.array([0] * len(df))

    @staticmethod
    def apply_sarima_detection(df, config):
        """Apply SARIMA model for anomaly detection"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            sarima_config = config.get('sarima', {})
            print(f" apply_sarima_detection() : inputs - config is : {config}")
            # Fit SARIMA model
            order = (sarima_config.get('p', 1), sarima_config.get('d', 1), sarima_config.get('q', 1)) 
            seasonal_order = (sarima_config.get('P',1), sarima_config.get('D',1), sarima_config.get('Q',1), sarima_config.get('s',12))
            sarima_threshold_value = sarima_config.get('threshold_value', 2.0)
            sarima_threshold_method = sarima_config.get('threshold_method', 'std')
        
            model = SARIMAX(df['value'], order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
        
            # Get residuals
            residuals = fitted_model.resid
        
            # Detect anomalies
            # Detect anomalies based on residual threshold
            threshold_method =  sarima_config.get('threshold_method', 'percentile')
            threshold_value = sarima_config.get('threshold_value', 2.0)

            # Apply threshold to get binary anomaly flags
            if threshold_method == 'percentile':
                threshold = np.percentile(residuals, 95)
            elif threshold_method == 'std':
                threshold = np.mean(residuals) + threshold_value * np.std(residuals)
            else:
                threshold = 0.1
        
            anomalies = np.abs(residuals) > threshold

            df['sarima'] = anomalies
            anomaly_scores = np.abs(residuals)
            df['sarima_anomaly_scores'] = anomaly_scores
        
            return anomalies, anomaly_scores
        
        except ImportError:
            print(f"statsmodels not installed. Run: pip install statsmodels")
            return np.array([False] * len(df)), np.array([0] * len(df))


    @staticmethod
    def apply_auto_arima_detection(df, config):
        """Apply auto_arima model for anomaly detection"""
        try:
            import pmdarima as pm

            aa_config = config.get('auto_arima', {})
            print(f" apply_arima_detection() : inputs - config is : {aa_config}")
        
            # Auto-Arima model
            model = pm.auto_arima(df['value'], 
                                 aa_config.get('seasonal', True),
                                 aa_config.get('stepwise', True),
                                 aa_config.get('suppress_warnings', True),
                                 aa_config.get('max_p', 5),
                                 aa_config.get('max_q', 5),
                                 error_action="ignore"
                                 )
            fitted_model = model.fit()
        
            # Get residuals
            residuals = fitted_model.resid
        
            # Detect anomalies based on residual threshold
            threshold_method =  aa_config.get('threshold_method', 'percentile')
            threshold_value = aa_config.get('threshold_value', 2.0)

            # Apply threshold to get binary anomaly flags
            if threshold_method == 'percentile':
                threshold = np.percentile(residuals, 95)
            elif threshold_method == 'std':
                threshold = np.mean(residuals) + threshold_value * np.std(residuals)
            else:
                threshold = 0.1
        
            anomalies = np.abs(residuals) > threshold

            df['auto_arima'] = anomalies
            anomaly_scores = np.abs(residuals)
            df['auto_arima_anomaly_scores'] = anomaly_scores
        
            return anomalies, anomaly_scores
        
        except ImportError:
            print(f"statsmodels not installed. Run: pip install statsmodels")
            return np.array([False] * len(df)), np.array([0] * len(df))


    @staticmethod
    def check_trend_seasonality(df):
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller

        # Check for Trend using ADF Test
        adf_result = adfuller(df['value'])
        has_trend = adf_result[1] <= 0.05
        print(f"ADF p-value: {adf_result[1]:.4f}")
        print(f"Trend detected: {has_trend}")

        # Check for Seasonality using Seasonal Decomposition
        decomposition = seasonal_decompose(df['value'], model='additive', period=12)
        seasonal_std = decomposition.seasonal.std()
        has_seasonality = seasonal_std > 0.5  # A simple heuristic, adjust threshold as needed
        print(f"Standard deviation of seasonal component: {seasonal_std:.4f}")
        print(f"Seasonality detected: {has_seasonality}")
        return has_trend, has_seasonality

    @staticmethod
    def apply_exponential_smoothing_detection(df, config):
        """Apply exponential_smoothing model for anomaly detection"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            es_config = config.get('exponential_smoothing', {})
            print(f" apply_exponential_smoothing_detection() : inputs - config is : {es_config}")

            has_trend, has_seasonality = MLBasedAnomaly.check_trend_seasonality(df)

            # Determine parameters based on our analysis
            trend_type = 'add' if has_trend else None
            seasonal_type = 'add' if has_seasonality else None
            seasonal_periods = 12 if has_seasonality else None

            # but we are getting these values from User . So let's use them for now. 
            trend_type = es_config.get('trend', 'add')
            seasonal_type = es_config.get('seasonal', 'add')
            seasonal_periods = es_config.get('seasonal_periods', 12)

             
            # ExponentialSmoothing model
            model = ExponentialSmoothing(df['value'], 
                                         trend = trend_type,
                                         seasonal = seasonal_type,
                                         seasonal_periods = seasonal_periods,
                                         initialization_method="estimated"
                                        )
            fitted_model = model.fit()
            print(f"Chosen model: ExponentialSmoothing(trend={trend_type}, seasonal={seasonal_type}, seasonal_periods={seasonal_periods})")
        
            # Get residuals
            residuals = fitted_model.resid
        
            # Detect anomalies based on residual threshold
            threshold_method =  es_config.get('threshold_method', 'percentile')
            threshold_value = es_config.get('threshold_value', 2.0)

            # Apply threshold to get binary anomaly flags
            if threshold_method == 'percentile':
                threshold = np.percentile(residuals, 95)
            elif threshold_method == 'std':
                threshold = np.mean(residuals) + threshold_value * np.std(residuals)
            else:
                threshold = 0.1
        
            anomalies = np.abs(residuals) > threshold

            df['exponential_smoothing'] = anomalies
            anomaly_scores = np.abs(residuals)
            df['exponential_smoothing_anomaly_scores'] = anomaly_scores
        
            return anomalies, anomaly_scores
        
        except ImportError:
            print(f"statsmodels not installed. Run: pip install statsmodels")
            return np.array([False] * len(df)), np.array([0] * len(df))


    @staticmethod
    def get_scaled_data (df):
        # Prepare features
        feature_columns = ['value', 'hour', 'day_of_week', 'hour_sin', 'hour_cos',
                          'day_sin', 'day_cos', 'week_of_year', 'month', 'day_of_month']

        # Add boolean features as integers
        bool_features = ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']
        for col in bool_features:
            if col in df.columns:
                feature_columns.append(col)

        # Create feature matrix
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()

        # Convert boolean columns to int
        for col in bool_features:
            if col in X.columns:
                X[col] = X[col].astype(int)

        # Handle any remaining NaN values
        X = X.fillna(X.mean())

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    @staticmethod
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
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
    
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler, available_features

    @staticmethod
    def apply_isolation_forest(df, config):
        try:
            if_config = config.get('isolation_forest', {}) 
            print(f" apply_isolation_forest() : inputs - config is : {if_config}")
            # Use enhanced feature scaling
            scaling_method = if_config.get('scaling_method', 'robust') 
            X_scaled, scaler, feature_names = MLBasedAnomaly.get_scaled_features_enhanced(df, scaling_method)

            # calculate approximate contamination
            actual_anomaly_rate = len(df[df.get('is_anomaly', False)]) / len(df) if 'is_anomaly' in df.columns else 0.041
            contamination = if_config.get('contamination', max(0.01, min(0.5, actual_anomaly_rate * 1.2)))
            n_estimators = if_config.get('n_estimators', 200)
            max_samples = if_config.get('max_samples', 'auto')
            random_state = if_config.get('random_state', 42)
            

            # Apply Isolation Forest

            # iso_forest = IsolationForest(contamination=contamination, random_state=42)
            # Enhanced Isolation Forest parameters
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,  # Increased for stability
                max_samples=max_samples,
                max_features=if_config.get('max_features', 0.8),  # Feature sampling
                random_state=random_state,
                bootstrap=True
            )

            # Get anomaly scores first
            anomaly_scores = -iso_forest.fit(X_scaled).score_samples(X_scaled)

            # Apply hybrid threshold approach
            predictions = iso_forest.predict(X_scaled)
            score_based_anomalies = predictions == -1
            
            # Add threshold-based detection for high values
            if config.get('use_threshold_hybrid', True):
                value_threshold = np.percentile(df['value'], config.get('value_percentile', 98))
                threshold_anomalies = df['value'] > value_threshold
            
                # Combine both approaches
                hybrid_anomalies = score_based_anomalies | threshold_anomalies
            else:
                hybrid_anomalies = score_based_anomalies

            print(f"Isolation Forest: {score_based_anomalies.sum()} score-based, {threshold_anomalies.sum() if 'threshold_anomalies' in locals() else 0} threshold-based")
            print(f"Total hybrid anomalies: {hybrid_anomalies.sum()}")
        
            # Store results
            df['isolation_forest'] = hybrid_anomalies
            df['isolation_forest_anomaly_scores'] = anomaly_scores
        
            return hybrid_anomalies, anomaly_scores

        except Exception as e:
            print(f"Error in apply_isolation_forest: {e}")
            print(f"Available columns in df: {df.columns.tolist()}")
            return np.zeros(len(df), dtype=bool), np.zeros(len(df))


    @staticmethod
    def apply_one_class_svm_detection(df, config):
        try:
            ocsvm_config = config.get('one_class_svm', {})
            print(f" apply_one_class_svm_detection() : inputs - config is : {ocsvm_config}")
            # X_scaled = self.get_scaled_data(df)
            X_scaled, scaler, feature_names = MLBasedAnomaly.get_scaled_features_enhanced(df, ocsvm_config.get('scaling_method', 'robust'))
            nu = ocsvm_config.get('nu', 0.1)
            kernel = ocsvm_config.get('kernel', 'rbf')
            gamma = ocsvm_config.get('gamma', 'scale')
            degree = ocsvm_config.get('degree', 3)

            svm = OneClassSVM(kernel = kernel, 
                              gamma = gamma,
                              nu=nu, 
                              degree = degree)
            predictions =  svm.fit_predict(X_scaled)

            anomalies = predictions == -1
            print(f"svm detected {anomalies.sum()} anomalies out of {len(anomalies)} points")
            anomaly_scores = svm.score_samples(X_scaled)

            df['one_class_svm'] = anomalies
            df['one_class_svm_anomaly_scores'] = anomaly_scores
            return anomalies, anomaly_scores
        except Exception as e:
            print(f"Error in apply_one_class_svm_detection: {e}")
            print(f"Available columns in df: {df.columns.tolist()}")
            return np.zeros(len(df), dtype=bool), np.zeros(len(df))



    @staticmethod
    def apply_dbscan_detection(df, config):
        try:
            dbscan_config = config.get('dbscan', {})
            print(f" apply_dbscan_detection() : inputs - config is : {dbscan_config}")
            # X_scaled = self.get_scaled_data(df)
            X_scaled, scaler, feature_names = MLBasedAnomaly.get_scaled_features_enhanced(df, dbscan_config.get('scaling_method', 'robust'))

            eps = dbscan_config.get('dbscan_eps', 0.5)
            min_samples = dbscan_config.get('dbscan_min_samples', 5)
            metric = dbscan_config.get('metric', 'euclidean')
            algorithm = dbscan_config.get('algorithm', 'auto')

            dbscan = DBSCAN(eps=eps, 
                            min_samples=min_samples,
                            metric =  metric,
                            algorithm = algorithm)

            clusters = dbscan.fit_predict(X_scaled)
            df['dbscan'] = clusters == -1  # Noise points are anomalies
            scores = np.zeros(len(clusters))
            anomalies = clusters == -1
            # Calculate scores only if there are anomalies
            if np.any(anomalies):
                scores = MLBasedAnomaly.calculate_dbscan_scores(X_scaled, clusters)
            else:
                scores = np.zeros(len(clusters))

            df['dbscan'] = anomalies
            df['dbscan_anomaly_scores'] = scores

            return anomalies, scores

        except Exception as e:
            print(f"Error in apply_dbscan_detection: {e}")
            print(f"Config: {config}")
            return np.zeros(len(df), dtype=bool), np.zeros(len(df))

    @staticmethod
    def apply_lof_detection(df, config):
        try:
            lof_config = config.get('local_outlier_factor', {})
            print(f" apply_lof_detection() : inputs - config is : {lof_config}")
            # Define feature columns (you need to define this)
            feature_cols = ['value', 'hour', 'day_of_week', 'hour_sin', 'hour_cos',
                           'day_sin', 'day_cos', 'week_of_year', 'month', 'day_of_month']
        
            # Add boolean features if they exist
            bool_features = ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']
            available_features = [col for col in feature_cols + bool_features if col in df.columns]


            feature_data = df[available_features].fillna(0)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)

            print(f"Feature data shape: {feature_data.shape}")
            print(f"Available features: {available_features}")
            print(f"Feature data stats:\n{feature_data.describe()}")
            print(f"Any NaN values: {feature_data.isna().sum().sum()}")


            n_neighbors = lof_config.get('n_neighbors', min(20, len(df)//2))
            contamination = lof_config.get('contamination', 0.03)
            algorithm = lof_config.get('algorithm','auto')
            leaf_size = lof_config.get('leaf_size', 30)
            if len(df) < n_neighbors + 1:
                print(f"Not enough data for LOF: need at least {n_neighbors + 1} samples, got {len(df)}")
                return np.zeros(len(df), dtype=bool), np.zeros(len(df))


            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(n_neighbors=n_neighbors, 
                                     contamination=contamination,
                                     algorithm = algorithm,
                                     leaf_size = leaf_size)

            # predictions = lof.fit_predict(feature_data)
            predictions = lof.fit_predict(feature_data_scaled)

            anomalies = predictions == -1  # Convert to boolean
            scores = lof.negative_outlier_factor_

            anomaly_indices = np.where(predictions == -1)[0]
            print(f"Anomaly indices: {anomaly_indices}")

            print(f"LOF predictions unique values: {np.unique(predictions, return_counts=True)}")
            print(f"LOF scores range: {scores.min():.4f} to {scores.max():.4f}")
            print(f"LOF scores mean: {scores.mean():.4f}")
            print(f"Expected anomalies with contamination {contamination}: {int(len(df) * contamination)}")

            df['local_outlier_factor'] = anomalies
            df['local_outlier_factor_anomaly_scores'] = scores

            return anomalies, scores 
        except Exception as e:
            print(f" Exception in lof execution : {e}")
            return np.zeros(len(df), dtype=bool), np.zeros(len(df))
            #method_reasons['lof'] = ["LOF: Error"] * len(df)


    @staticmethod
    def apply_usad_detection(df, config):
        print(f"\n=== USAD DEBUG INFO ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"Config received: {config}")

        try:
            usad_config = config.get('usad', {})
            print(f" apply_usad_detection() : inputs - config is : {usad_config}")
            # Define feature columns
            feature_cols = ['value']  # Start with just value column
        
            # Add time-based features if available
            time_features = ['hour', 'day_of_week', 'hour_sin', 'hour_cos',
                            'day_sin', 'day_cos', 'week_of_year', 'month', 'day_of_month']
            bool_features = ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']


            # Check which features exist
            available_features = []
            for col in feature_cols + time_features + bool_features:
                if col in df.columns:
                    available_features.append(col)
                
            print(f"Available features: {available_features}")
            if not available_features:
                print("ERROR: No valid features found!")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            # Prepare data
            X = df[available_features].copy()
            print(f"Feature data shape: {X.shape}")
            print(f"Feature data info:")
            print(X.describe())

            # Check for missing values
            missing_count = X.isnull().sum().sum()
            print(f"Missing values: {missing_count}")
        
            # Fill missing values and convert to numpy
            X_filled = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            X_numpy = X_filled.to_numpy(dtype=np.float32)

            print(f"Final input shape: {X_numpy.shape}")
            print(f"Input data range: [{X_numpy.min():.4f}, {X_numpy.max():.4f}]")
            print(f"Input data mean: {X_numpy.mean():.4f}, std: {X_numpy.std():.4f}")
        
            # Check if data has variance
            if X_numpy.std() < 1e-6:
                print("WARNING: Input data has very low variance!")

            # Get device
            device = get_torch_device()
            print(f"Using device: {device}")
        
            # Train USAD model with debugging
            print("\n=== TRAINING USAD ===")
            try:
                usad_model, usad_scaler = train_usad(
                    X_numpy, 
                    latent_dim=usad_config.get('hidden_size', 32), 
                    epochs=usad_config.get('epochs', 50),  # Increased default epochs
                    batch_size=usad_config.get('batch_size', 64), 
                    device=device
                )
                print("USAD training completed successfully")
            except Exception as e:
                print(f"ERROR in USAD training: {str(e)}")
                import traceback
                traceback.print_exc()
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            # Get anomaly scores with debugging
            print("\n=== GETTING USAD SCORES ===")
            try:
                anomaly_scores = usad_score(usad_model, usad_scaler, X_filled, device=device)
                print(f"Scores computed successfully, shape: {anomaly_scores.shape}")
            except Exception as e:
                print(f"ERROR in USAD scoring: {str(e)}")
                import traceback
                traceback.print_exc()
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)
        
            # Detailed score analysis
            print(f"\n=== SCORE ANALYSIS ===")
            print(f"Raw scores shape: {anomaly_scores.shape}")
            print(f"Raw scores type: {type(anomaly_scores)}")
            print(f"Raw scores - min: {anomaly_scores.min():.8f}")
            print(f"Raw scores - max: {anomaly_scores.max():.8f}")
            print(f"Raw scores - mean: {anomaly_scores.mean():.8f}")
            print(f"Raw scores - std: {anomaly_scores.std():.8f}")
            print(f"Raw scores - median: {np.median(anomaly_scores):.8f}")
        
            # Check for NaN or infinite values
            nan_count = np.isnan(anomaly_scores).sum()
            inf_count = np.isinf(anomaly_scores).sum()
            print(f"NaN scores: {nan_count}, Infinite scores: {inf_count}")

        
            if nan_count > 0 or inf_count > 0:
                print("WARNING: Invalid scores detected, replacing with 0")
                anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
            # Show score distribution
            percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]
            print(f"Score percentiles:")
            for p in percentiles:
                val = np.percentile(anomaly_scores, p)
                print(f"  {p}th percentile: {val:.8f}")
        
            # Try multiple threshold methods
            print(f"\n=== THRESHOLD TESTING ===")
        
            # Method 1: Percentile-based (most common)
            threshold_value = config.get('threshold_value', 0.05)  # Default 5% anomalies
            percentile = (1 - threshold_value) * 100
            threshold_percentile = np.percentile(anomaly_scores, percentile)
            anomalies_pct = anomaly_scores > threshold_percentile
            print(f"Percentile method ({percentile:.1f}th): threshold={threshold_percentile:.8f}, anomalies={anomalies_pct.sum()}")
        
            # Method 2: Standard deviation
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            threshold_std = mean_score + 2 * std_score
            anomalies_std = anomaly_scores > threshold_std
            print(f"Std method (mean + 2*std): threshold={threshold_std:.8f}, anomalies={anomalies_std.sum()}")

            # Method 3: IQR method
            Q1 = np.percentile(anomaly_scores, 25)
            Q3 = np.percentile(anomaly_scores, 75)
            IQR = Q3 - Q1
            threshold_iqr = Q3 + 1.5 * IQR
            anomalies_iqr = anomaly_scores > threshold_iqr
            print(f"IQR method (Q3 + 1.5*IQR): threshold={threshold_iqr:.8f}, anomalies={anomalies_iqr.sum()}")
        
            # Method 4: Fixed percentage of highest scores
            top_n_percent = 0.05  # Top 5%
            n_anomalies = max(1, int(len(anomaly_scores) * top_n_percent))
            threshold_topn = np.partition(anomaly_scores, -n_anomalies)[-n_anomalies]
            anomalies_topn = anomaly_scores >= threshold_topn
            print(f"Top-N method (top {top_n_percent*100}%): threshold={threshold_topn:.8f}, anomalies={anomalies_topn.sum()}")
        
            # Choose the best method
            threshold_method = config.get('threshold_method', 'percentile')
        
            if threshold_method == 'percentile':
                final_threshold = threshold_percentile
                final_anomalies = anomalies_pct
            elif threshold_method == 'std':
                final_threshold = threshold_std
                final_anomalies = anomalies_std
            elif threshold_method == 'iqr':
                final_threshold = threshold_iqr
                final_anomalies = anomalies_iqr
            else:  # top_n
                final_threshold = threshold_topn
                final_anomalies = anomalies_topn

            # If still no anomalies, force some detection
            if final_anomalies.sum() == 0:
                print("\n=== FORCING ANOMALY DETECTION ===")
                print("No anomalies detected with any method, forcing top 1% as anomalies")
                n_forced = max(1, int(len(anomaly_scores) * 0.01))  # At least 1% or 1 point
                final_threshold = np.partition(anomaly_scores, -n_forced)[-n_forced]
                final_anomalies = anomaly_scores >= final_threshold
                print(f"Forced detection: threshold={final_threshold:.8f}, anomalies={final_anomalies.sum()}")
        
            print(f"\n=== FINAL RESULTS ===")
            print(f"Method used: {threshold_method}")
            print(f"Final threshold: {final_threshold:.8f}")
            print(f"Final anomalies: {final_anomalies.sum()} out of {len(final_anomalies)}")
            print(f"Anomaly percentage: {(final_anomalies.sum()/len(final_anomalies))*100:.2f}%")
        
            # Store results in dataframe
            df["usad_anomaly_scores"] = anomaly_scores
            df['usad'] = final_anomalies
        
            # Show some example anomalies
            if final_anomalies.sum() > 0:
                anomaly_indices = np.where(final_anomalies)[0]
                print(f"Example anomaly indices: {anomaly_indices[:5]}")
                print(f"Example anomaly scores: {anomaly_scores[anomaly_indices[:5]]}")
        
            return final_anomalies, anomaly_scores
        
        except Exception as e:
            print(f"\n=== CRITICAL ERROR IN USAD ===")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
            # Return default values
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    # Additional helper function to check your USAD implementation
    @staticmethod
    def debug_usad_model(df):
        """Quick test to verify USAD model is working"""
        print("=== USAD MODEL TEST ===")
    
        # Create simple test data
        test_data = np.random.normal(0, 1, (100, 2)).astype(np.float32)
        # Add some clear outliers
        test_data[95:] = np.random.normal(5, 1, (5, 2)).astype(np.float32)
    
        print(f"Test data shape: {test_data.shape}")
        print(f"Normal data mean: {test_data[:95].mean():.4f}")
        print(f"Outlier data mean: {test_data[95:].mean():.4f}")
    
        try:
            device = get_torch_device()
            usad_model, usad_scaler = train_usad(test_data, latent_dim=16, epochs=10, batch_size=32, device=device)
        
            test_df = pd.DataFrame(test_data, columns=['feature1', 'feature2'])
            scores = usad_score(usad_model, usad_scaler, test_df, device=device)
        
            print(f"Test scores shape: {scores.shape}")
            print(f"Normal scores mean: {scores[:95].mean():.4f}")
            print(f"Outlier scores mean: {scores[95:].mean():.4f}")
            print(f"Score ratio (outlier/normal): {scores[95:].mean() / scores[:95].mean():.2f}")
        
            if scores[95:].mean() > scores[:95].mean():
                print("✅ USAD model is working - outliers have higher scores")
            else:
                print("❌ USAD model issue - outliers don't have higher scores")
            
        except Exception as e:
            print(f"❌ USAD model test failed: {str(e)}")
            import traceback
            traceback.print_exc()


    @staticmethod
    def apply_elliptic_envelope_detection(df, config):
        """Apply Elliptic Envelope anomaly detection"""
        try:
            elliptic_envelope_config = config.get('elliptic_envelope', {})
            print(f" apply_elliptic_envelope_detection() : inputs - config is : {elliptic_envelope_config}")
            from sklearn.covariance import EllipticEnvelope
        
            # Get config values
            contamination = elliptic_envelope_config.get('contamination', 0.1)
            support_fraction = elliptic_envelope_config.get('support_fraction', 0.8)
            random_state = elliptic_envelope_config.get('random_state', 42)
        
            # Prepare features - use same feature engineering as other models
            feature_cols = ['value']
        
            # Add time-based features if available
            time_features = ['hour', 'day_of_week', 'hour_sin', 'hour_cos',
                            'day_sin', 'day_cos', 'week_of_year', 'month', 'day_of_month']
            bool_features = ['is_weekend', 'is_peak_hour', 'is_off_hour', 'is_holiday']
        
            available_features = [col for col in feature_cols + time_features + bool_features 
                                if col in df.columns]
        
            X = df[available_features].fillna(0)
        
            print(f"Elliptic Envelope using features: {available_features}")
            print(f"Elliptic Envelope config - contamination: {contamination}, support_fraction: {support_fraction}")
        
            # Create and fit model
            model = EllipticEnvelope(
                contamination=contamination,
                support_fraction=support_fraction,
                random_state=random_state
            )
        
            # Fit and predict (-1 for anomaly, 1 for normal)
            predictions = model.fit_predict(X)
            anomalies = predictions == -1  # Convert to boolean
        
            # Get anomaly scores (distance from fitted ellipse)
            scores = model.decision_function(X)
            # Convert to positive scores (higher = more anomalous)
            anomaly_scores = -scores
        
            print(f"Elliptic Envelope anomalies detected: {anomalies.sum()} out of {len(anomalies)}")
            print(f"Elliptic Envelope score range: {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")
        
            # Store results
            df['elliptic_envelope_scores'] = anomaly_scores
            df['elliptic_envelope'] = anomalies
        
            return anomalies, anomaly_scores
        
        except Exception as e:
            print(f"Error in Elliptic Envelope detection: {str(e)}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def calculate_ensemble_scores(model_scores, method='weighted_average'):
        """Calculate ensemble scores based on selected method"""
        if method == 'weighted_average':
            total_weight = sum(score_data['weight'] for score_data in model_scores.values())
            ensemble = np.zeros(len(next(iter(model_scores.values()))['scores']))
        
            for model_name, score_data in model_scores.items():
                weight_ratio = score_data['weight'] / total_weight
                ensemble += score_data['scores'] * weight_ratio
            
        elif method == 'top_n':
            # Implementation for top-N ensemble
            pass
    
        return ensemble

    @staticmethod
    def apply_darts_thresholds(anomaly_scores, threshold_method, threshold_value, percentile_threshold):
        """Apply threshold logic to convert Darts anomaly scores to binary flags"""
    
        if threshold_method == 'percentile':
            # Use percentile_threshold (e.g., 95th percentile)
            threshold = np.percentile(anomaly_scores, percentile_threshold)
        
        elif threshold_method == 'std':
            # Use threshold_value as number of standard deviations
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            threshold = mean_score + threshold_value * std_score
        
        elif threshold_method == 'mad':
            # Use threshold_value with Modified Absolute Deviation
            median_score = np.median(anomaly_scores)
            mad = np.median(np.abs(anomaly_scores - median_score))
            threshold = median_score + threshold_value * mad * 1.4826
        
        elif threshold_method == 'fixed':
            # Use threshold_value directly as fixed threshold
            threshold = threshold_value
        
        else:
            # Default: use percentile_threshold
            threshold = np.percentile(anomaly_scores, percentile_threshold)

        anomalies = anomaly_scores > threshold
    
        return anomalies

    @staticmethod
    def convert_to_darts_ts_simple(df, time_col_name=None):
        """
        Converts a pandas DataFrame to a scaled Darts TimeSeries.
    
        This function is simplified to only handle conversion and scaling,
        as slicing and filtering should be done in the main function.
        """
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
        if not numerical_cols:
            print("No numerical columns found. Skipping.")
            return None

        df[numerical_cols] = df[numerical_cols].astype('float32')

        try:

            # Step 1: Prepare the DataFrame with a proper time index
            if time_col_name and time_col_name in df.columns:
                # Ensure the time column is in a proper datetime format
                df[time_col_name] = pd.to_datetime(df[time_col_name])
                df = df.set_index(time_col_name).sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                print("Warning: No valid time column or DatetimeIndex found. Cannot determine frequency.")
                return None
        
            # Step 2: Dynamically determine the frequency
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is None:
                time_diffs = df.index.to_series().diff().dropna()
                # Use the most common difference
                mode_diff = time_diffs.mode()
                if not mode_diff.empty:
                    inferred_freq = pd.tseries.frequencies.to_offset(mode_diff[0]).freqstr


            if inferred_freq is None:
                print("Could not infer a consistent frequency from the data.")
                # Default to a safe, small frequency if all else fails
                inferred_freq = '5min' 

            # Step 3: Create Darts TimeSeries
            ts = TimeSeries.from_dataframe(
                df,
                time_col=None, # Index is now the time axis
                value_cols=numerical_cols, 
                freq=inferred_freq, 
                fill_missing_dates=True
            )


            filler = MissingValuesFiller()
            ts = filler.transform(ts)
        
            if len(ts) == 0 or np.any(np.isnan(ts.values())):
                print("TimeSeries is empty or contains NaNs after filling. Skipping.")
                return None

            # Scale the data using a new scaler each time
            scaler = Scaler()
            scaled_ts = scaler.fit_transform(ts)

            return scaled_ts

        except Exception as e:
            print(f"Conversion to Darts TimeSeries failed: {e}")
            return None


    @staticmethod
    def convert_to_darts_ts(df, model_type_name, sequence_length, ts_for_training=None, validation_split=None):

        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numerical_cols:
            print(f"{model_type_name} skipped: No numerical columns found for {model_type_name}.")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

        try:
            # Convert to Darts TimeSeries (multivariate)
            ts = TimeSeries.from_dataframe( df[numerical_cols], time_col=None, value_cols=numerical_cols, freq='5min', fill_missing_dates=True )
            filler = MissingValuesFiller()
            ts = filler.transform(ts)

            if len(ts) == 0 or np.any(np.isnan(ts.values())):
                print(f"{model_type_name}: Initial TimeSeries is empty or contains NaNs after filling. Skipping.")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            # Scale the data
            scaler = Scaler()
            
            data_len = len(ts) # Use unscaled ts length for chunk calculation
            chunk_length = min(sequence_length, max(1, data_len // 10)) # Minimum 1

            if data_len < chunk_length * 2:
                print(f"Data length ({data_len}) too short for {model_type_name} with chunk_length ({chunk_length}). Falling back.")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            # Use ts_for_training if provided (for calibration), otherwise use a slice of the full ts
            if ts_for_training is not None:
                print(f"Using provided training time series of length {len(ts_for_training)}")
                # Convert DataFrame to TimeSeries if needed
                if isinstance(ts_for_training, pd.DataFrame):
                    ts_to_train_model_on = TimeSeries.from_dataframe(
                        ts_for_training[numerical_cols], time_col=None, 
                        value_cols=numerical_cols, freq='5min', fill_missing_dates=True
                    )
                    ts_to_train_model_on = filler.transform(ts_to_train_model_on)
                else:
                    ts_to_train_model_on = ts_for_training
                    print(f"ts_to_train_model_on = ts_for_training is executed ")

            else:
                train_size = int((1 - validation_split) * len(ts))
                ts_to_train_model_on = ts[:max(chunk_length * 2, train_size)]
                print(f"Using {len(ts_to_train_model_on)} points for training (validation_split={validation_split})")
            
            #if len(ts_to_train_model_on) == 0 or np.any(np.isnan(ts_to_train_model_on.values.flatten())):
            #if len(ts_to_train_model_on) == 0 or np.any(pd.isna(ts_to_train_model_on.values.flatten())):
            if len(ts_to_train_model_on) == 0 :
                print(f"{model_type_name}: Training data is empty or contains NaNs.")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            # Fit scaler on the (potentially filtered) training data
            print(f"fit_transform()")
            scaled_train_ts = scaler.fit_transform(ts_to_train_model_on)
            # Transform the full original series for prediction
            print(f"transform()")
            scaled_ts_for_prediction = scaler.transform(ts)

            #if len(scaled_train_ts) == 0 or np.any(np.isnan(scaled_train_ts.values.flatten())):
            #if len(scaled_train_ts) == 0 or pd.any(pd.isna(scaled_train_ts.values.flatten())):
            if len(scaled_train_ts) == 0 :
                print(f"{model_type_name}: Scaled training series is empty or contains NaNs. Skipping training.")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            #if len(scaled_ts_for_prediction) == 0 or np.any(np.isnan(scaled_ts_for_prediction.values.flatten())):
            #if len(scaled_ts_for_prediction) == 0 or pd.any(pd.isna(scaled_ts_for_prediction.values.flatten())):
            if len(scaled_ts_for_prediction) == 0 :
                print(f"{model_type_name}: Scaled prediction series is empty or contains NaNs. Skipping prediction.")
                return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

            print(f"return from convert_to_darts_ts()")
            return scaled_train_ts, scaled_ts_for_prediction

        except Exception as e:
            print(f"{model_type_name} anomaly detection failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)


    @staticmethod
    def apply_lstm_forecasting_detection_v1 (df, config):

        lstm_config = config.get('lstm', {})
        hidden_dim = lstm_config.get('hidden_size', 64)
        n_rnn_layers = lstm_config.get('num_layers',2)
        input_chunk_length = lstm_config.get('sequence_length', 24)
        n_epochs = lstm_config.get('epochs', 50)
        batch_size = lstm_config.get('batch_size', 32)
        learning_rate = lstm_config.get('learning_rate', 0.001)

        threshold_method = lstm_config.get('threshold_method', 'Percentile')
        threshold_value = lstm_config.get('threshold_value', 0.05)
        percentile_threshold = lstm_config.get('percentile_threshold', 95.0)
        validation_split = lstm_config.get('validation_split', 0.2)

        model_type_name =  "lstm"


        metrics = df.select_dtypes(include="number").columns.tolist()
        col = metrics[0]

        ts_all = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols=[col])
        target_ts = ts_all
        target_ts = target_ts.astype(np.float32)

        # Create RNN model with automatic encoders
        rnn = RNNModel(
            model="LSTM" ,
            input_chunk_length=24,
            output_chunk_length=1,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={'lr': learning_rate},
            force_reset=True,
            add_encoders={
                    'cyclic': {'future': ['hour']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']}
            },
            random_state=42
        )

        try:
            # Fit using only target series (encoders auto-add covariates)
            rnn.fit(series=target_ts)
            horizon = len(target_ts)

            # Predict into the future without manual covariates
            pr = rnn.predict(n=horizon, series=target_ts)

            resid = np.abs(target_ts.values().flatten() - pr.values().flatten())
            anomalies = MLBasedAnomaly.apply_darts_thresholds(resid, threshold_method, threshold_value, percentile_threshold)

            # Store results
            df['lstm_scores'] = resid
            df['lstm'] = anomalies

            return anomalies, resid

        except Exception as e:
            print(f"{model_type_name} anomaly detection failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_lstm_forecasting_detection_v2 (df, config, ts_for_training = None):

        try:
            lstm_config = config.get('lstm', {})
            print(f" apply_lstm_forecasting_detection() : inputs - config is : {lstm_config}")

            hidden_dim = lstm_config.get('hidden_size', 64)
            n_rnn_layers = lstm_config.get('num_layers',2)
            input_chunk_length = lstm_config.get('sequence_length', 50)
            n_epochs = lstm_config.get('epochs', 50)
            batch_size = lstm_config.get('batch_size', 32)
            learning_rate = lstm_config.get('learning_rate', 0.001)

            threshold_method = lstm_config.get('threshold_method', 'Percentile')
            threshold_value = lstm_config.get('threshold_value', 0.05)
            percentile_threshold = lstm_config.get('percentile_threshold', 95.0)
            validation_split = lstm_config.get('validation_split', 0.2)

            model_type_name =  "lstm"
            # --- CRITICAL CHANGE 1: Filter normal data for training ---
            # Assuming 'is_anomaly' is a boolean column in your DataFrame.
            if 'is_anomaly' in df.columns:
                train_df = df[df['is_anomaly'] == False].copy()
            else:
                # Fallback to 80:20 split if no labels exist
                train_size = int(0.8 * len(df))
                train_df = df.iloc[:train_size].copy()

            # Convert and scale the training and full series separately
            scaled_train_ts = MLBasedAnomaly.convert_to_darts_ts_simple(train_df, 'timestamp')
            scaled_train_ts_for_prediction = MLBasedAnomaly.convert_to_darts_ts_simple(df, 'timestamp')

            training_length = len(scaled_train_ts)

            # Check if training data is sufficient
            output_chunk_length = 1
            if training_length < input_chunk_length + output_chunk_length:
                print(f"Training data length ({training_length}) is too short for the configured `input_chunk_length` ({input_chunk_length}).")
                # Reduce `input_chunk_length` to a safe, small value if the data is insufficient
                new_input_chunk_length = max(1, training_length - output_chunk_length)
                print(f"Adjusting `input_chunk_length` to {new_input_chunk_length}.")
                input_chunk_length = new_input_chunk_length


            lstm_training_length = training_length - 1

            lstm_model = RNNModel(
                model='LSTM',
                hidden_dim=hidden_dim,
                n_rnn_layers=n_rnn_layers,
                input_chunk_length=input_chunk_length,
                output_chunk_length=1,  # For anomaly detection
                n_epochs=n_epochs,
                batch_size=batch_size,
                optimizer_kwargs={'lr': learning_rate},
                random_state=42,
                force_reset=True,
                training_length=lstm_training_length 
            )
    
            # Train the model
            print(f"before fit : training_length is : {training_length}")
            lstm_model.fit(scaled_train_ts) # Fit on potentially filtered and scaled training data


            # Generate predictions and calculate errors
            prediction_errors = []

            print(f" input_chunk_length : {input_chunk_length} and len(scaled_train_ts_for_prediction) : {len(scaled_train_ts_for_prediction)}")

            # Make rolling one-step-ahead predictions on the full series
            for i in range(input_chunk_length, len(scaled_train_ts_for_prediction)):
                try:
                    # Use a single, correctly-named variable for the model
                    hist_data = scaled_train_ts_for_prediction[i - input_chunk_length : i]
        
                    # Predict the next time step
                    pred = lstm_model.predict(n=output_chunk_length, series=hist_data)
        
                    # Get the actual value for that time step
                    actual = scaled_train_ts_for_prediction[i]

                    if pred is None or len(pred) == 0 or np.any(np.isnan(pred.values())):
                        print(f"{model_type_name}: Prediction in loop resulted in empty series or NaNs at index {i}. Appending 0.0 error.")
                        prediction_errors.append(0.0)
                        continue

                    # Calculate the error between the one-step-ahead prediction and the actual value
                    error = np.abs(actual.values().flatten()[0] - pred.values().flatten()[0])
                    prediction_errors.append(error)

                except Exception as inner_e:
                    print(f"Inner {model_type_name} loop failed at index {i}: {inner_e}. Appending 0.0 error.")
                    prediction_errors.append(0.0)


            # Pad errors to match dataframe length
            full_errors = [0.0] * input_chunk_length # Pad beginning with zeros as no prediction possible
            full_errors.extend(prediction_errors)
            full_errors.extend([0.0] * (len(df) - len(full_errors)))
            full_errors = full_errors[:len(df)]

            anomaly_scores = pd.Series(full_errors, index=df.index) # Return raw errors

            anomalies = MLBasedAnomaly.apply_darts_thresholds(anomaly_scores, threshold_method, threshold_value, percentile_threshold)

            # Store results
            df['lstm_scores'] = anomaly_scores
            df['lstm'] = anomalies

            return anomalies, anomaly_scores

        except Exception as e:
            print(f"{model_type_name} anomaly detection failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_lstm_forecasting_detection(df, config, ts_for_training=None):
        """Improved LSTM-based anomaly detection with batch processing"""
        try:
            lstm_config = config.get('lstm', {})
            print(f"LSTM Config: {lstm_config}")
            
            # Model parameters
            hidden_dim = lstm_config.get('hidden_size', 64)
            n_rnn_layers = lstm_config.get('num_layers', 2)
            input_chunk_length = lstm_config.get('sequence_length', 50)
            n_epochs = lstm_config.get('epochs', 100)  # Increased default
            batch_size = lstm_config.get('batch_size', 32)
            learning_rate = lstm_config.get('learning_rate', 0.001)
            
            # Threshold parameters
            threshold_method = lstm_config.get('threshold_method', 'dynamic_percentile')
            percentile_threshold = lstm_config.get('percentile_threshold', 95.0)
            
            # Prepare training data with improved strategy
            train_df, train_size = _prepare_training_data(df)
            
            # Data preprocessing with robust scaling
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            
            # Fit scaler on training data only
            train_values = train_df['value'].values.reshape(-1, 1)
            scaler.fit(train_values)
            
            # Scale both training and full data
            scaled_train_values = scaler.transform(train_values).flatten()
            scaled_full_values = scaler.transform(df['value'].values.reshape(-1, 1)).flatten()
            
            # Create DataFrames with scaled values for Darts conversion
            train_df_scaled = train_df.copy()
            train_df_scaled['value'] = scaled_train_values
            
            full_df_scaled = df.copy()
            full_df_scaled['value'] = scaled_full_values
            
            # Convert to Darts TimeSeries using your existing method
            train_ts = MLBasedAnomaly.convert_to_darts_ts_simple(train_df_scaled, 'timestamp')
            full_ts = MLBasedAnomaly.convert_to_darts_ts_simple(full_df_scaled, 'timestamp')
            
            # Detect and prepare datetime encoders
            encoders = _detect_datetime_features(df, 'timestamp')
            print(f"Using datetime encoders: {encoders}")
            
            # Check data sufficiency
            if len(train_ts) < input_chunk_length + 10:  # Need some buffer
                input_chunk_length = max(10, len(train_ts) // 2)
                print(f"Adjusted input_chunk_length to {input_chunk_length}")
            
            # Enhanced LSTM model with regularization and datetime encoders
            model_kwargs = {
                'model': 'LSTM',
                'hidden_dim': hidden_dim,
                'n_rnn_layers': n_rnn_layers,
                'input_chunk_length': input_chunk_length,
                'output_chunk_length': 1,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'dropout': 0.2,  # Add dropout for regularization
                'optimizer_kwargs': {'lr': learning_rate, 'weight_decay': 1e-5},
                'random_state': 42,
                'force_reset': True,
                'pl_trainer_kwargs': {
                    'enable_progress_bar': False,
                    'enable_checkpointing': False
                }
            }
            
            # Add encoders if available
            if encoders:
                model_kwargs['add_encoders'] = encoders
                print("Added datetime encoders to LSTM model")
            
            lstm_model = RNNModel(**model_kwargs)
            
            # Train model
            print(f"Training LSTM on {len(train_ts)} points...")
            lstm_model.fit(train_ts)
            
            # Improved batch prediction approach
            prediction_errors = []
            window_size = input_chunk_length
            stride = max(1, window_size // 4)  # Overlapping windows
            
            print(f"Generating predictions with stride {stride}...")
            
            # Process in batches for efficiency
            for start_idx in range(window_size, len(full_ts), stride):
                try:
                    # Get historical window
                    hist_window = full_ts[start_idx - window_size:start_idx]
                    
                    # Predict next point
                    pred = lstm_model.predict(n=1, series=hist_window)
                    
                    if pred is not None and len(pred) > 0:
                        # Get actual value
                        if start_idx < len(full_ts):
                            actual_value = full_ts.values().flatten()[start_idx]
                            predicted_value = pred.values().flatten()[0] 
                            
                            # Calculate prediction error
                            error = float(np.abs(actual_value - predicted_value))
                            
                            # Fill stride positions with same error
                            for _ in range(min(stride, len(full_ts) - start_idx)):
                                prediction_errors.append(error)
                        
                except Exception as e:
                    print(f"Prediction error at index {start_idx}: {e}")
                    # Fill with zeros for failed predictions
                    for _ in range(min(stride, len(full_ts) - start_idx)):
                        prediction_errors.append(0.0)
            
            # Ensure correct length
            full_errors = [0.0] * window_size  # Pad beginning
            full_errors.extend(prediction_errors)
            full_errors = full_errors[:len(df)]
            
            # Add padding if needed
            while len(full_errors) < len(df):
                full_errors.append(np.mean(prediction_errors) if prediction_errors else 0.0)
            
            # Create anomaly scores
            anomaly_scores = pd.Series(full_errors, index=df.index)
            
            # Apply improved threshold
            threshold = _adaptive_threshold(
                anomaly_scores.values, threshold_method, percentile_threshold
            )
            
            anomalies = anomaly_scores > threshold
            
            print(f"LSTM: Detected {anomalies.sum()}/{len(df)} anomalies with threshold {threshold:.6f}")
            
            # Store results
            df['lstm_scores'] = anomaly_scores
            df['lstm'] = anomalies
            
            return anomalies, anomaly_scores
            
        except Exception as e:
            print(f"LSTM anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)


    @staticmethod
    def apply_gru_forecasting_detection(df, config, ts_for_training=None):
        """Improved GRU-based anomaly detection with batch processing"""
        try:
            gru_config = config.get('gru', {})
            print(f"GRU Config: {gru_config}")
            
            # Model parameters
            hidden_dim = gru_config.get('hidden_size', 64)
            n_rnn_layers = gru_config.get('num_layers', 2)
            input_chunk_length = gru_config.get('sequence_length', 50)
            n_epochs = gru_config.get('epochs', 100)  # Increased default
            batch_size = gru_config.get('batch_size', 32)
            learning_rate = gru_config.get('learning_rate', 0.001)
            
            # Threshold parameters
            threshold_method = gru_config.get('threshold_method', 'dynamic_percentile')
            percentile_threshold = gru_config.get('percentile_threshold', 95.0)
            
            # Prepare training data with improved strategy
            train_df, train_size = _prepare_training_data(df)
            
            # Data preprocessing with robust scaling
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            
            # Fit scaler on training data only
            train_values = train_df['value'].values.reshape(-1, 1)
            scaler.fit(train_values)
            
            # Scale both training and full data
            scaled_train_values = scaler.transform(train_values).flatten()
            scaled_full_values = scaler.transform(df['value'].values.reshape(-1, 1)).flatten()
            
            # Create DataFrames with scaled values for Darts conversion
            train_df_scaled = train_df.copy()
            train_df_scaled['value'] = scaled_train_values
            
            full_df_scaled = df.copy()
            full_df_scaled['value'] = scaled_full_values
            
            # Convert to Darts TimeSeries using your existing method
            train_ts = MLBasedAnomaly.convert_to_darts_ts_simple(train_df_scaled, 'timestamp')
            full_ts = MLBasedAnomaly.convert_to_darts_ts_simple(full_df_scaled, 'timestamp')
            
            # Detect and prepare datetime encoders
            encoders = _detect_datetime_features(df, 'timestamp')
            print(f"Using datetime encoders: {encoders}")
            
            # Check data sufficiency
            if len(train_ts) < input_chunk_length + 10:  # Need some buffer
                input_chunk_length = max(10, len(train_ts) // 2)
                print(f"Adjusted input_chunk_length to {input_chunk_length}")
            
            # Enhanced LSTM model with regularization and datetime encoders
            model_kwargs = {
                'model': 'GRU',
                'hidden_dim': hidden_dim,
                'n_rnn_layers': n_rnn_layers,
                'input_chunk_length': input_chunk_length,
                'output_chunk_length': 1,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'dropout': 0.2,  # Add dropout for regularization
                'optimizer_kwargs': {'lr': learning_rate, 'weight_decay': 1e-5},
                'random_state': 42,
                'force_reset': True,
                'pl_trainer_kwargs': {
                    'enable_progress_bar': False,
                    'enable_checkpointing': False
                }
            }
            
            # Add encoders if available
            if encoders:
                model_kwargs['add_encoders'] = encoders
                print("Added datetime encoders to LSTM model")
            
            gru_model = RNNModel(**model_kwargs)
            
            # Train model
            print(f"Training GRU on {len(train_ts)} points...")
            gru_model.fit(train_ts)
            
            # Improved batch prediction approach
            prediction_errors = []
            window_size = input_chunk_length
            stride = max(1, window_size // 4)  # Overlapping windows
            
            print(f"Generating predictions with stride {stride}...")
            
            # Process in batches for efficiency
            for start_idx in range(window_size, len(full_ts), stride):
                try:
                    # Get historical window
                    hist_window = full_ts[start_idx - window_size:start_idx]
                    
                    # Predict next point
                    pred = gru_model.predict(n=1, series=hist_window)
                    
                    if pred is not None and len(pred) > 0:
                        # Get actual value
                        if start_idx < len(full_ts):
                            actual_value = full_ts.values().flatten()[start_idx]
                            predicted_value = pred.values().flatten()[0] 
                            
                            # Calculate prediction error
                            error = float(np.abs(actual_value - predicted_value))
                            
                            # Fill stride positions with same error
                            for _ in range(min(stride, len(full_ts) - start_idx)):
                                prediction_errors.append(error)
                        
                except Exception as e:
                    print(f"Prediction error at index {start_idx}: {e}")
                    # Fill with zeros for failed predictions
                    for _ in range(min(stride, len(full_ts) - start_idx)):
                        prediction_errors.append(0.0)
            
            # Ensure correct length
            full_errors = [0.0] * window_size  # Pad beginning
            full_errors.extend(prediction_errors)
            full_errors = full_errors[:len(df)]
            
            # Add padding if needed
            while len(full_errors) < len(df):
                full_errors.append(np.mean(prediction_errors) if prediction_errors else 0.0)
            
            # Create anomaly scores
            anomaly_scores = pd.Series(full_errors, index=df.index)
            
            # Apply improved threshold
            threshold = _adaptive_threshold(
                anomaly_scores.values, threshold_method, percentile_threshold
            )
            
            anomalies = anomaly_scores > threshold
            
            print(f"GRU: Detected {anomalies.sum()}/{len(df)} anomalies with threshold {threshold:.6f}")
            
            # Store results
            df['gru_scores'] = anomaly_scores
            df['gru'] = anomalies
            
            return anomalies, anomaly_scores
            
        except Exception as e:
            print(f"GRU anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_autoencoder_detection_v1(df, config, ts_for_training=None):
        """
        Apply RNN Autoencoder for anomaly detection with config integration
    
        Args:
            df: DataFrame with time series data
            config: Configuration dictionary with autoencoder parameters
            ts_for_training: Optional training time series (for calibration)
    
        Returns:
            pd.Series: Reconstruction errors for anomaly detection
        """

        model_type_name = "RNN Autoencoder"
        try:
            ae_config = config.get('autoencoder', {})
            print(f" apply_autoencoder_detection() : inputs - config is : {ae_config}")

            # Extract config parameters with defaults
            sequence_length = ae_config.get('sequence_length', 24)  # Use config or default
            encoding_dim = ae_config.get('encoding_dimension', 50)
            hidden_layers = ae_config.get('hidden_layers', [100, 50])
            activation = ae_config.get('activation_function', 'relu')
            epochs = ae_config.get('epochs', 100)
            batch_size = ae_config.get('batch_size', 32)
            learning_rate = ae_config.get('learning_rate', 0.001)
            validation_split = ae_config.get('validation_split', 0.2)
            threshold_method = ae_config.get('threshold_method', 'Percentile')
            threshold_value = ae_config.get('threshold_value', 0.05)
            percentile_threshold = ae_config.get('percentile_threshold', 95.0)
        
            chunk_length = sequence_length

            if ts_for_training is None:
                # Split the dataframe 80:20 for training
                train_size = int(0.8 * len(df))
                train_df = df.iloc[:train_size]
    
                scaled_train_ts, _ = MLBasedAnomaly.convert_to_darts_ts(train_df, model_type_name, sequence_length, ts_for_training, validation_split=validation_split)
                _, scaled_ts_for_prediction = MLBasedAnomaly.convert_to_darts_ts(df, model_type_name, sequence_length, ts_for_training, validation_split=validation_split)
            else:
                scaled_train_ts, scaled_ts_for_prediction = MLBasedAnomaly.convert_to_darts_ts(df, model_type_name, sequence_length, ts_for_training=ts_for_training, validation_split=validation_split)
            print(f"Using config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

            hidden_dim = hidden_layers[0] if hidden_layers else encoding_dim
            n_rnn_layers = len(hidden_layers) if len(hidden_layers) > 1 else 2

            activation_map = {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid'}
            darts_activation = activation_map.get(activation, 'ReLU')

            print(f"Model config: hidden_dim={hidden_dim}, layers={n_rnn_layers}, activation={darts_activation}")

            # Create RNN model with autoencoder-like configuration
            model = RNNModel( # Use RNNModel here and specify 'LSTM' for autoencoder
                model='LSTM', # Using LSTM for autoencoder as it's common for this task
                input_chunk_length=chunk_length,
                output_chunk_length=chunk_length,  # Same as input for reconstruction
                hidden_dim=hidden_dim, 
                n_rnn_layers=n_rnn_layers,
                dropout=0.1,
                n_epochs=epochs,
                batch_size=batch_size,
                optimizer_kwargs={'lr': learning_rate},
                random_state=42,
                force_reset=True,
                pl_trainer_kwargs={'accelerator': 'cpu', 'enable_progress_bar': False, 'max_epochs': epochs,}
            )

            print(f"Training autoencoder for {epochs} epochs...")

            # Train the model
            model.fit(scaled_train_ts) # Fit on potentially filtered and scaled training data

            print("Training completed. Calculating reconstruction errors...")

            # Calculate reconstruction errors
            reconstruction_errors = []

            # Process in overlapping windows on the full scaled series
            step_size = max(1, chunk_length // 8)
            for i in range(0, len(scaled_ts_for_prediction) - chunk_length + 1, step_size):
                try:
                    window = scaled_ts_for_prediction[i:i + chunk_length]
                    if len(window) == chunk_length:
                        # Use the trained model to reconstruct
                        pred = model.predict(n=chunk_length, series=window)

                        if pred is None or len(pred) == 0 or np.any(np.isnan(pred.values())):
                            print(f"{model_type_name}: Prediction in loop resulted in empty series or NaNs at index {i}. Appending 0.0 error.")
                            reconstruction_errors.extend([0.0] * step_size)
                            continue

                        actual_vals = window.values().flatten()
                        pred_vals = pred.values().flatten()

                        if len(actual_vals) == len(pred_vals):
                            error = np.mean(np.abs(actual_vals - pred_vals))
                            reconstruction_errors.extend([error] * step_size)
                        else:
                            print(f"{model_type_name}: Mismatch in length or empty actual/pred in window at index {i}. Appending 0.0 error.")
                            reconstruction_errors.extend([0.0] * step_size)
                    else:
                        print(f"{model_type_name}: Window length mismatch at index {i}. Appending 0.0 error.")
                        reconstruction_errors.extend([0.0] * step_size)
                except Exception as inner_e:
                    print(f"Inner {model_type_name} loop failed at index {i}: {inner_e}. Appending 0.0 error.")
                    reconstruction_errors.extend([0.0] * step_size)

            # Adjust length to match dataframe
            while len(reconstruction_errors) < len(df):
                reconstruction_errors.append(np.mean(reconstruction_errors) if reconstruction_errors else 0.0)
            reconstruction_errors = reconstruction_errors[:len(df)]

            print(f"Generated {len(reconstruction_errors)} reconstruction errors")

            reconstruction_series  = pd.Series(reconstruction_errors, index=df.index) # Return raw errors

            anomaly_predictions = MLBasedAnomaly.apply_darts_thresholds(reconstruction_series, threshold_method, threshold_value, percentile_threshold)
            print(f"Detected {anomaly_predictions.sum()} anomalies out of {len(df)} points")

            # Store results
            df['autoencoder_scores'] = reconstruction_series
            df['autoencoder'] = anomaly_predictions
        
            return anomaly_predictions, reconstruction_series  # Return raw errors for further processing


        except Exception as e:
            print(f"{model_type_name} failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_autoencoder_detection(df, config, ts_for_training=None):
        """Improved autoencoder-based anomaly detection using encoder-decoder architecture"""
        try:
            ae_config = config.get('autoencoder', {})
            print(f"Autoencoder Config: {ae_config}")
            
            # Model parameters
            sequence_length = ae_config.get('sequence_length', 24)
            encoding_dim = ae_config.get('encoding_dimension', 16)  # Smaller for bottleneck
            epochs = ae_config.get('epochs', 150)  # More epochs for autoencoder
            batch_size = ae_config.get('batch_size', 32)
            learning_rate = ae_config.get('learning_rate', 0.0005)  # Lower learning rate
            
            # Threshold parameters
            threshold_method = ae_config.get('threshold_method', 'dynamic_percentile')
            percentile_threshold = ae_config.get('percentile_threshold', 95.0)
            
            # Prepare training data
            train_df, train_size = _prepare_training_data(df)
            
            # Robust scaling
            scaler = RobustScaler()
            train_values = train_df['value'].values.reshape(-1, 1)
            scaler.fit(train_values)
            
            # Scale data
            scaled_train_values = scaler.transform(train_values).flatten()
            scaled_full_values = scaler.transform(df['value'].values.reshape(-1, 1)).flatten()
            
            # Create sequences for autoencoder training
            def create_sequences(data, seq_length):
                sequences = []
                for i in range(len(data) - seq_length + 1):
                    sequences.append(data[i:i + seq_length])
                return np.array(sequences)
            
            # Check data sufficiency
            if len(scaled_train_values) < sequence_length * 2:
                sequence_length = max(5, len(scaled_train_values) // 3)
                print(f"Adjusted sequence_length to {sequence_length}")
            
            # Create training sequences
            train_sequences = create_sequences(scaled_train_values, sequence_length)
            
            # Use RNNModel configured as an autoencoder
            # Note: This is a workaround - ideally use a dedicated autoencoder
            autoencoder_model = RNNModel(
                model='LSTM',
                input_chunk_length=sequence_length,
                output_chunk_length=sequence_length,  # Reconstruct the input
                hidden_dim=encoding_dim * 2,  # Encoder dimension
                n_rnn_layers=2,
                dropout=0.1,
                n_epochs=epochs,
                batch_size=batch_size,
                optimizer_kwargs={'lr': learning_rate, 'weight_decay': 1e-4},
                random_state=42,
                force_reset=True,
                pl_trainer_kwargs={
                    'enable_progress_bar': False,
                    'enable_checkpointing': False
                }
            )
            
            # Create DataFrames with scaled values for Darts conversion
            train_df_scaled = train_df.copy()
            train_df_scaled['value'] = scaled_train_values
            
            full_df_scaled = df.copy()
            full_df_scaled['value'] = scaled_full_values
            
            # Convert to Darts format using your existing method
            train_ts = MLBasedAnomaly.convert_to_darts_ts_simple(train_df_scaled, 'timestamp')
            full_ts = MLBasedAnomaly.convert_to_darts_ts_simple(full_df_scaled, 'timestamp')
            
            print(f"Training autoencoder for {epochs} epochs...")
            autoencoder_model.fit(train_ts)
            
            # Calculate reconstruction errors
            reconstruction_errors = []
            stride = max(1, sequence_length // 4)
            
            print("Calculating reconstruction errors...")
            
            for start_idx in range(0, len(full_ts) - sequence_length + 1, stride):
                try:
                    # Get sequence window
                    sequence = full_ts[start_idx:start_idx + sequence_length]
                    
                    if len(sequence) == sequence_length:
                        # Get reconstruction
                        reconstruction = autoencoder_model.predict(
                            n=sequence_length, 
                            series=sequence
                        )
                        
                        if reconstruction is not None and len(reconstruction) == sequence_length:
                            # Calculate mean absolute error for reconstruction
                            # Correctly flatten the values() to get proper arrays
                            original_values = sequence.values().flatten()
                            reconstructed_values = reconstruction.values().flatten()
                            
                            # Calculate reconstruction error as a single float
                            mse = float(np.mean((original_values - reconstructed_values) ** 2))
                            
                            # Assign error to all points in stride
                            for _ in range(min(stride, len(full_ts) - start_idx)):
                                reconstruction_errors.append(mse)
                        else:
                            # Fill with zeros for failed reconstructions
                            for _ in range(min(stride, len(full_ts) - start_idx)):
                                reconstruction_errors.append(0.0)
                    
                except Exception as e:
                    print(f"Reconstruction error at index {start_idx}: {e}")
                    for _ in range(min(stride, len(full_ts) - start_idx)):
                        reconstruction_errors.append(0.0)
            
            # Ensure correct length
            while len(reconstruction_errors) < len(df):
                reconstruction_errors.append(
                    np.mean(reconstruction_errors) if reconstruction_errors else 0.0
                )
            reconstruction_errors = reconstruction_errors[:len(df)]
            
            # Create anomaly scores
            anomaly_scores = pd.Series(reconstruction_errors, index=df.index)
            
            # Apply improved threshold
            threshold = _adaptive_threshold(
                anomaly_scores.values, threshold_method, percentile_threshold
            )
            
            anomalies = anomaly_scores > threshold
            
            print(f"Autoencoder: Detected {anomalies.sum()}/{len(df)} anomalies with threshold {threshold:.6f}")
            
            # Store results
            df['autoencoder_scores'] = anomaly_scores
            df['autoencoder'] = anomalies
            
            return anomalies, anomaly_scores
            
        except Exception as e:
            print(f"Autoencoder anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)



    @staticmethod
    def calculate_adaptive_threshold(scores, method='percentile', percentile=95):
        """Calculate adaptive threshold for anomaly detection"""
        if method == 'percentile':
            return np.percentile(scores, percentile)
        elif method == 'iqr':
            Q3 = np.percentile(scores, 75)
            Q1 = np.percentile(scores, 25)
            IQR = Q3 - Q1
            return Q3 + 1.5 * IQR
        elif method == 'mad':  # Median Absolute Deviation
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            return median + 3 * mad

    @staticmethod
    def calculate_dbscan_scores(X_scaled, clusters):
        """Calculate anomaly scores for DBSCAN results"""
        from sklearn.neighbors import NearestNeighbors
    
        scores = np.zeros(len(clusters))
    
        # For noise points (anomalies), calculate distance to nearest cluster
        if len(np.unique(clusters[clusters != -1])) > 0:
            # Find cluster centers
            cluster_centers = []
            for cluster_id in np.unique(clusters[clusters != -1]):
                cluster_points = X_scaled[clusters == cluster_id]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
        
            if cluster_centers:
                cluster_centers = np.array(cluster_centers)
            
                # Calculate distances for anomalies
                anomaly_indices = np.where(clusters == -1)[0]
                for idx in anomaly_indices:
                    point = X_scaled[idx].reshape(1, -1)
                    distances = np.linalg.norm(cluster_centers - point, axis=1)
                    scores[idx] = np.min(distances)  # Distance to nearest cluster
    
        return scores

    @staticmethod
    def apply_zscore_detection(df, config):
        try:    
            model_type_name = 'zscore'
            rule_config = config.get('zscore', {})  # Fixed: was config('zscore', {})
        
            threshold = rule_config.get('threshold', 3.0)
            use_modified = rule_config.get('use_modified', False)
            window_size = rule_config.get('window_size', 30)
        
            if use_modified:
                # Rolling Modified Z-Score
                if window_size and window_size > 1:
                    # Rolling median and MAD
                    rolling_median = df['value'].rolling(window=window_size, center=True).median()
                    rolling_mad = df['value'].rolling(window=window_size, center=True).apply(
                        lambda x: np.median(np.abs(x - np.median(x)))
                    )
                    # Handle edge cases where MAD is 0
                    rolling_mad = rolling_mad.replace(0, np.nan).fillna(rolling_mad.median())
                
                    modified_z_scores = 0.6745 * (df['value'] - rolling_median) / rolling_mad
                    anomalies = np.abs(modified_z_scores) > threshold
                    scores = np.abs(modified_z_scores)
                else:
                    # Global Modified Z-Score (original logic)
                    median = df['value'].median()
                    mad = np.median(np.abs(df['value'] - median))
                    if mad == 0:
                        mad = df['value'].std()  # Fallback if MAD is 0
                    modified_z_scores = 0.6745 * (df['value'] - median) / mad
                    anomalies = np.abs(modified_z_scores) > threshold
                    scores = np.abs(modified_z_scores)
            else:
                # Standard Z-Score
                if window_size and window_size > 1:
                    # Rolling Z-Score
                    rolling_mean = df['value'].rolling(window=window_size, center=True).mean()
                    rolling_std = df['value'].rolling(window=window_size, center=True).std()
                    # Handle edge cases where std is 0
                    rolling_std = rolling_std.replace(0, np.nan).fillna(rolling_std.median())
                
                    z_scores = np.abs((df['value'] - rolling_mean) / rolling_std)
                    anomalies = z_scores > threshold
                    scores = z_scores
                else:
                    # Global Z-Score (original logic)
                    z_scores = np.abs(stats.zscore(df['value']))
                    anomalies = z_scores > threshold
                    scores = z_scores
        
            # Handle NaN values from rolling calculations
            anomalies = anomalies.fillna(False)
            scores = scores.fillna(0)
        
            df['zscore'] = anomalies
            df['zscore_anomaly_scores'] = scores
        
            return anomalies, scores
        
        except Exception as e:
            print(f"{model_type_name} failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_mad_detection(df, config):
        try:
            model_type_name = 'mad'
            rule_config = config.get('mad', {})
        
            threshold = rule_config.get('threshold', 3.0)
            window_size = rule_config.get('window_size', 30)
            constant = rule_config.get('constant', 1.4826)  # Default constant for normal distribution
        
            if window_size and window_size > 1:
                # Rolling MAD
                rolling_median = df['value'].rolling(window=window_size, center=True).median()
                rolling_mad = df['value'].rolling(window=window_size, center=True).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                # Handle edge cases where MAD is 0
                rolling_mad = rolling_mad.replace(0, np.nan).fillna(rolling_mad.median())
            
                # Calculate modified Z-scores using rolling statistics
                mad_scores = constant * np.abs(df['value'] - rolling_median) / rolling_mad
                anomalies = mad_scores > threshold
                scores = mad_scores
            else:
                # Global MAD (traditional approach)
                median = df['value'].median()
                mad = np.median(np.abs(df['value'] - median))
                if mad == 0:
                    mad = df['value'].std() * 0.6745  # Fallback
            
                mad_scores = constant * np.abs(df['value'] - median) / mad
                anomalies = mad_scores > threshold
                scores = mad_scores
        
            # Handle NaN values from rolling calculations
            anomalies = anomalies.fillna(False)
            scores = scores.fillna(0)
        
            df['mad'] = anomalies
            df['mad_anomaly_scores'] = scores
        
            return anomalies, scores
        
        except Exception as e:
            print(f"{model_type_name} failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)

    @staticmethod
    def apply_ewma_detection(df, config):
        try:
            model_type_name = 'ewma'
            rule_config = config.get('ewma', {})
        
            alpha = rule_config.get('alpha', 0.3)
            threshold = rule_config.get('threshold', 3.0)
            lambda_param = rule_config.get('lambda_param', 0.2)  # For variance calculation
            startup_periods = rule_config.get('startup_periods', 10)

            print(f"EWMA config: alpha={alpha}, threshold={threshold}, lambda_param={lambda_param}, startup_periods={startup_periods}")

        
            values = df['value'].values
            n = len(values)
        
            # Initialize EWMA mean and variance
            ewma_mean = np.zeros(n)
            ewma_var = np.zeros(n)
            ewma_std = np.zeros(n) # Initialize ewma_std as well 


            # Initialize with first value
            ewma_mean[0] = values[0]
            ewma_var[0] = 0
        
            # Calculate EWMA mean and variance
            for i in range(1, n):
                # EWMA mean
                ewma_mean[i] = alpha * values[i] + (1 - alpha) * ewma_mean[i-1]
            
                # EWMA variance (using lambda parameter)
                residual = values[i] - ewma_mean[i-1]
                ewma_var[i] = lambda_param * (residual ** 2) + (1 - lambda_param) * ewma_var[i-1]
        
                # Calculate EWMA standard deviation and score inside the loop
                ewma_std[i] = np.sqrt(ewma_var[i])

                # Handle zero standard deviation
                if ewma_std[i] == 0:
                    ewma_std[i] = np.std(values[:i+1]) # Or some other small constant

                ewma_score = np.abs(values[i] - ewma_mean[i]) / ewma_std[i]

                # Print key values for each time step
                print(f"Index: {i}")
                print(f"  Current Value: {values[i]:.2f}")
                print(f"  EWMA Mean: {ewma_mean[i]:.2f}")
                print(f"  EWMA Std: {ewma_std[i]:.2f}") # This will be after you calculate ewma_std
                print(f"  Upper Control Limit: {ewma_mean[i] + threshold * ewma_std[i]:.2f}")
                print(f"  Lower Control Limit: {ewma_mean[i] - threshold * ewma_std[i]:.2f}")
                print(f"  EWMA Score: {ewma_score:.2f}")
                print("-" * 20)

            # Calculate EWMA standard deviation
            ewma_std = np.sqrt(ewma_var)
        
            # Handle startup periods and zero variance
            ewma_std = np.where(ewma_std == 0, np.std(values[:startup_periods+1]), ewma_std)
        
            # Debug: Check EWMA calculations
            print(f"EWMA mean range: {ewma_mean.min():.3f} to {ewma_mean.max():.3f}")
            print(f"EWMA std range: {ewma_std.min():.3f} to {ewma_std.max():.3f}")
            print(f"Value range: {df['value'].min():.3f} to {df['value'].max():.3f}")

            # First, calculate the scores and handle NaNs directly
            ewma_scores = np.abs(values - ewma_mean) / ewma_std
            ewma_scores = np.nan_to_num(ewma_scores, nan=0.0) # Replaces NaNs with 0.0

            # Then, create the anomalies boolean array
            # This line is correct and already handles the arange part
            anomalies = (ewma_scores > threshold) & (np.arange(n) >= startup_periods)

            # No need for fillna() on anomalies since it's a boolean array
            # But to be safe, you can convert it to boolean again
            anomalies = anomalies.astype(bool)

            # The rest of your code to add to df
            df['ewma'] = anomalies
            df['ewma_anomaly_scores'] = ewma_scores

            return anomalies, ewma_scores
        
        except Exception as e:
            print(f"{model_type_name} failed: {e}")
            return pd.Series([False] * len(df), index=df.index), pd.Series([0.0] * len(df), index=df.index)


    @staticmethod
    def apply_ml_based_detection(df, models, config):
        print(f" apply_ml_based_detection() : inputs - config is : {config}")
        """Apply ML detection with scoring and ensemble methods"""
        results = {}
        scores = {}
    
        # Calculate scores for each selected model
        for model_name, model_config in models.items():
            print(f"In apply_ml_based_detection() : model_name is : {model_name}")
            if not model_config['enabled']:
                print(f"In apply_ml_based_detection() : {model_name} is not enabled ")
                continue

            if model_config['enabled'] and config.get(model_name, False):
                if model_name == 'prophet':
                    anomalies, model_scores = MLBasedAnomaly.apply_prophet_detection(df, config)
                elif model_name == 'arima':
                    anomalies, model_scores = MLBasedAnomaly.apply_arima_detection(df, config)
                elif model_name == 'sarima':
                    anomalies, model_scores = MLBasedAnomaly.apply_sarima_detection(df, config)
                elif model_name == 'auto_arima':
                    anomalies, model_scores = MLBasedAnomaly.apply_auto_arima_detection(df, config)
                elif model_name == 'exponential_smoothing':
                    anomalies, model_scores = MLBasedAnomaly.apply_exponential_smoothing_detection(df, config)
                elif model_name == 'isolation_forest':
                    anomalies, model_scores = MLBasedAnomaly.apply_isolation_forest(df, config)
                elif model_name == 'one_class_svm':
                    anomalies, model_scores = MLBasedAnomaly.apply_one_class_svm_detection(df, config)
                elif model_name == 'dbscan':
                    anomalies, model_scores = MLBasedAnomaly.apply_dbscan_detection(df, config)
                elif model_name == 'kmeans':
                    anomalies, model_scores = apply_kmeans_detection(df, config)
                elif model_name == 'local_outlier_factor':
                    anomalies, model_scores = MLBasedAnomaly.apply_lof_detection(df, config)
                elif model_name == 'usad':
                    anomalies, model_scores = MLBasedAnomaly.apply_usad_detection(df, config)
                elif model_name == 'elliptic_envelope':
                    anomalies, model_scores = MLBasedAnomaly.apply_elliptic_envelope_detection(df, config)
                elif model_name == 'lstm':
                    anomalies, model_scores = MLBasedAnomaly.apply_lstm_forecasting_detection(df, config)
                    # anomalies, model_scores = MLBasedAnomaly.apply_lstm_simple(df, config)
                elif model_name == 'gru':
                    anomalies, model_scores = MLBasedAnomaly.apply_gru_forecasting_detection(df, config)
                elif model_name == 'autoencoder':
                    anomalies, model_scores = MLBasedAnomaly.apply_autoencoder_detection(df, config)
                elif model_name == 'zscore':
                    anomalies, model_scores = MLBasedAnomaly.apply_zscore_detection(df,config)
                elif model_name == 'mad':
                    anomalies, model_scores = MLBasedAnomaly.apply_mad_detection(df,config)
                elif model_name == 'ewma':
                    anomalies, model_scores = MLBasedAnomaly.apply_ewma_detection(df,config)

                else:
                    print(f"In apply_ml_based_detection() : {model_name} is not executed ")
                    continue

                print(f"\n=== {model_name.upper()} ANOMALIES ===")
                print(f"Total anomalies: {anomalies.sum()}")
                results[model_name] = anomalies
                scores[model_name] = {
                    'scores': model_scores,
                    'weight': model_config['weight']
                }
    
        # Calculate ensemble scores
        if 'ensemble_method' in config and 'threshold_method' in config: 
            ensemble_scores = MLBasedAnomaly.calculate_ensemble_scores(scores, config['ensemble_method'])
    
            # Apply adaptive threshold
            threshold = MLBasedAnomaly.calculate_adaptive_threshold(ensemble_scores, config['threshold_method'])
            results['ensemble'] = ensemble_scores > threshold
    
        # return results, scores, ensemble_scores
        return results


