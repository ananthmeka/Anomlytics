
# ENHANCED MODEL-PROPERTIES MAPPING
# ===================================

# RULES-BASED METHODS
# -------------------
RULES_MAPPING = {
    'static_min_threshold': {
        'conditions': [
            {'condition': 'domain_knowledge_available == True', 'severity': 'must', 'weight': 30},
            {'condition': 'has_negative == False', 'severity': 'optional', 'weight': 10},
            {'condition': 'anomaly_ratio_estimate < 0.05', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 85,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'static_max_threshold': {
        'conditions': [
            {'condition': 'domain_knowledge_available == True', 'severity': 'must', 'weight': 30},
            {'condition': 'skew < 1.0', 'severity': 'optional', 'weight': 10},
            {'condition': 'anomaly_ratio_estimate < 0.05', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 85,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'static_range_rule': {
        'conditions': [
            {'condition': 'domain_knowledge_available == True', 'severity': 'must', 'weight': 30},
            {'condition': 'size >= 500', 'severity': 'optional', 'weight': 15},
            {'condition': 'anomaly_ratio_estimate < 0.15', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 80,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'percentile_rule': {
        'conditions': [
            {'condition': 'domain_knowledge_available == False', 'severity': 'must', 'weight': 20},
            {'condition': 'skew > 1.0 or kurtosis > 3.5', 'severity': 'must', 'weight': 30},
            {'condition': 'anomaly_ratio_estimate > 0.02', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 65,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'static_percentage_rule': {
        'conditions': [
            {'condition': 'domain_knowledge_available == True', 'severity': 'must', 'weight': 25},
            {'condition': 'seasonal_strength > 0.4', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 75,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'sharp_jump_rule': {
        'conditions': [
            {'condition': 'roll_kurt_10 > 5 or high_pass_10 > 0.7', 'severity': 'must', 'weight': 40},
            {'condition': 'noise_level < 0.3', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 70,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'rate_of_change': {
        'conditions': [
            {'condition': 'trend_ratio > 0.3', 'severity': 'must', 'weight': 25},
            {'condition': 'autocorrelation_score > 0.5', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 60,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'trend_rule': {
        'conditions': [
            {'condition': 'trend_ratio > 0.4', 'severity': 'must', 'weight': 25},
            {'condition': 'is_stationary == False', 'severity': 'must', 'weight': 30}
        ],
        'base_priority': 75,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'baseline_deviation_rule': {
        'conditions': [
            {'condition': 'trend_ratio > 0.6', 'severity': 'must', 'weight': 20},
            {'condition': 'noise_level < 0.5', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 70,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'seasonal_decomp_rule': {
        'conditions': [
            {'condition': 'seasonal_strength > 0.6', 'severity': 'must', 'weight': 30},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 25},
            {'condition': 'size >= 2 * dominant_period', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 80,
        'computational_cost': 'MEDIUM',
        'real_time_capable': False,
        'interpretable': True
    },
    'periodic_rule': {
        'conditions': [
            {'condition': 'autocorrelation_score > 0.5', 'severity': 'must', 'weight': 25},
            {'condition': 'dominant_period >= 4 and dominant_period <= 365', 'severity': 'must', 'weight': 20}
        ],
        'base_priority': 70,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'consecutive_anomaly': {
        'conditions': [
            {'condition': 'outlier_persistence > 1.0', 'severity': 'must', 'weight': 25}
        ],
        'base_priority': 65,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
}

ML_MAPPING = {
    'isolation_forest': {
        'conditions': [
            {'condition': 'n_features > 1', 'severity': 'must', 'weight': 20},
            {'condition': 'size >= 2000', 'severity': 'optional', 'weight': 15},
            {'condition': 'anomaly_ratio_estimate > 0.05', 'severity': 'optional', 'weight': 10},
            {'condition': 'missing_value_ratio < 0.15', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 80,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': False
    },
    'one_class_svm': {
        'conditions': [
            {'condition': 'n_features <= 50', 'severity': 'must', 'weight': 25},
            {'condition': 'size >= 1000', 'severity': 'optional', 'weight': 15},
            {'condition': 'missing_value_ratio < 0.1', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 65,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': False
    },
    'elliptic_envelope': {
        'conditions': [
            {'condition': 'n_features <= 20', 'severity': 'must', 'weight': 20},
            {'condition': 'skew >= -0.5 and skew <= 0.5', 'severity': 'must', 'weight': 30},
            {'condition': 'kurtosis < 5', 'severity': 'must', 'weight': 25},
            {'condition': 'size >= 500', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 60,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': False
    },
    'local_outlier_factor': {
        'conditions': [
            {'condition': 'size >= 200', 'severity': 'optional', 'weight': 10},
            {'condition': 'n_features <= 50', 'severity': 'must', 'weight': 20},
            {'condition': 'anomaly_ratio_estimate > 0.03', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 70,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': False
    },
    'dbscan': {
        'conditions': [
            {'condition': 'n_features >= 2', 'severity': 'must', 'weight': 15},
            {'condition': 'size >= 1000', 'severity': 'optional', 'weight': 10},
            {'condition': 'anomaly_ratio_estimate > 0.05', 'severity': 'optional', 'weight': 10},
            {'condition': 'has_density_based_clusters == True', 'severity': 'must', 'weight': 50}
        ],
        'base_priority': 75,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': False
    },
    'kmeans': {
        'conditions': [
            {'condition': 'n_features >= 2', 'severity': 'must', 'weight': 15},
            {'condition': 'size >= 1000', 'severity': 'optional', 'weight': 10},
            {'condition': 'has_density_based_clusters == False', 'severity': 'must', 'weight': 30}
        ],
        'base_priority': 65,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': False
    },
    'zscore': {
        'conditions': [
            {'condition': 'size >= 50', 'severity': 'optional', 'weight': 5},
            {'condition': 'skew >= -0.5 and skew <= 0.5', 'severity': 'must', 'weight': 45},
            {'condition': 'kurtosis <= 3.5', 'severity': 'must', 'weight': 40},
            {'condition': 'anomaly_ratio_estimate < 0.05', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 60,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'mad': {
        'conditions': [
            {'condition': 'size >= 50', 'severity': 'optional', 'weight': 5},
            {'condition': 'skew > 0.5 or kurtosis > 3.5', 'severity': 'optional', 'weight': 20},
            {'condition': 'anomaly_ratio_estimate < 0.15', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 65,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'ewma': {
        'conditions': [
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
            {'condition': 'trend_ratio < 0.3', 'severity': 'must', 'weight': 15}
        ],
        'base_priority': 70,
        'computational_cost': 'LOW',
        'real_time_capable': True,
        'interpretable': True
    },
    'prophet': {
        'conditions': [
            {'condition': 'n_features == 1', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 25},
            {'condition': 'seasonal_strength > 0.6', 'severity': 'must', 'weight': 30},
            {'condition': 'dominant_period >= 7 and dominant_period <= 365', 'severity': 'must', 'weight': 25},
            {'condition': 'size >= 2 * dominant_period', 'severity': 'must', 'weight': 20},
            {'condition': 'trend_ratio > 0.4', 'severity': 'optional', 'weight': 10},
            {'condition': 'missing_value_ratio < 0.2', 'severity': 'optional', 'weight': 5}
        ],
        'base_priority': 90,
        'computational_cost': 'HIGH',
        'real_time_capable': False,
        'interpretable': True
    },
    'arima': {
        'conditions': [
            {'condition': 'n_features == 1', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 25},
            {'condition': 'is_stationary == True', 'severity': 'must', 'weight': 40},
            {'condition': 'autocorrelation_score > 0.4', 'severity': 'must', 'weight': 30},
            {'condition': 'size >= 200', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 85,
        'computational_cost': 'HIGH',
        'real_time_capable': False,
        'interpretable': True
    },
    'sarima': {
        'conditions': [
            {'condition': 'n_features == 1', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 25},
            {'condition': 'is_stationary == False', 'severity': 'must', 'weight': 30},
            {'condition': 'seasonal_strength > 0.4', 'severity': 'must', 'weight': 35},
            {'condition': 'autocorrelation_score > 0.5', 'severity': 'must', 'weight': 30},
            {'condition': 'size >= 200', 'severity': 'optional', 'weight': 10}
        ],
        'base_priority': 85,
        'computational_cost': 'HIGH',
        'real_time_capable': False,
        'interpretable': True
    },
    'exponential_smoothing': {
        'conditions': [
            {'condition': 'n_features == 1', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
            {'condition': 'seasonal_strength > 0.4', 'severity': 'optional', 'weight': 15}
        ],
        'base_priority': 75,
        'computational_cost': 'MEDIUM',
        'real_time_capable': True,
        'interpretable': True
    },
    'lstm': {
        'conditions': [
            {'condition': 'n_features > 1', 'severity': 'optional', 'weight': 5},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
            {'condition': 'size >= 5000', 'severity': 'must', 'weight': 50},
            {'condition': 'autocorrelation_score > 0.6', 'severity': 'must', 'weight': 40}
        ],
        'base_priority': 90,
        'computational_cost': 'VERY_HIGH',
        'real_time_capable': False,
        'interpretable': False
    },
    'gru': {
        'conditions': [
            {'condition': 'n_features > 1', 'severity': 'optional', 'weight': 5},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
            {'condition': 'size >= 5000', 'severity': 'must', 'weight': 50},
            {'condition': 'autocorrelation_score > 0.6', 'severity': 'must', 'weight': 40}
        ],
        'base_priority': 90,
        'computational_cost': 'VERY_HIGH',
        'real_time_capable': False,
        'interpretable': False
    },
    'autoencoder': {
        'conditions': [
            {'condition': 'size >= 10000', 'severity': 'must', 'weight': 50},
            {'condition': 'n_features >= 2', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
        ],
        'base_priority': 85,
        'computational_cost': 'VERY_HIGH',
        'real_time_capable': False,
        'interpretable': False
    },
    'usad': {
        'conditions': [
            {'condition': 'size >= 10000', 'severity': 'must', 'weight': 50},
            {'condition': 'n_features >= 2', 'severity': 'must', 'weight': 20},
            {'condition': 'is_regular == True', 'severity': 'must', 'weight': 20},
            {'condition': 'noise_level < 0.5', 'severity': 'optional', 'weight': 10},
        ],
        'base_priority': 95,
        'computational_cost': 'VERY_HIGH',
        'real_time_capable': False,
        'interpretable': False
    },
}



'''
RULES_MAPPING = {
    # Threshold-based rules
    'static_min_threshold': {
        'conditions': [
            'domain_knowledge_available == True',  # New flag needed
            'has_negative == False',  # Min thresholds make sense for positive data
            'anomaly_ratio_estimate < 0.05'  # Works best with few anomalies
        ],
        'priority': 'HIGH'  # Domain knowledge should have high priority
    },
    
    'static_max_threshold': {
        'conditions': [
            'domain_knowledge_available == True',
            'anomaly_ratio_estimate < 0.15',  # Not too many anomalies
            'skew < 1.5'  # Not highly skewed
        ],
        'priority': 'HIGH'
    },
    
    'static_range_rule': {
        'conditions': [
            'domain_knowledge_available == True',
            'anomaly_ratio_estimate < 0.10',
            'size >= 500'  # Need sufficient data to validate range
        ],
        'priority': 'HIGH'
    },
    
    # Statistical rules
    'percentile_rule': {
        'conditions': [
            'skew > 1.0 or kurtosis > 3.5',  # Non-normal distributions
            'anomaly_ratio_estimate < 0.15',
            'size >= 100'
        ],
        'priority': 'MEDIUM'
    },
    
    'static_percentage_rule': {
        'conditions': [
            'anomaly_ratio_estimate > 0.05',  # When you expect more anomalies
            'trend_ratio < 0.3',  # Stable baseline
            'seasonal_strength < 0.3'  # No strong seasonality
        ],
        'priority': 'MEDIUM'
    },
    
    # Pattern-based rules  
    'sharp_jump_rule': {
        'conditions': [
            'roll_kurt_10 > 5 or high_pass_10 > 0.7',  # Spiky signals
            'anomaly_ratio_estimate > 0.02',  # At least 2% anomalies expected
            'noise_level < 0.5'  # Not too noisy
        ],
        'priority': 'HIGH'
    },
    
    'rate_of_change': {
        'conditions': [
            'trend_ratio > 0.3',  # Has trending behavior
            'autocorrelation_score > 0.4',  # Some temporal dependence
            'size >= 200'
        ],
        'priority': 'MEDIUM'
    },
    
    'trend_rule': {
        'conditions': [
            'trend_ratio > 0.4',  # Strong trend required
            'is_stationary == False',  # Non-stationary data
            'size >= 200'
        ],
        'priority': 'HIGH'
    },
    
    'baseline_deviation_rule': {
        'conditions': [
            'trend_ratio < 0.3',  # Stable baseline needed
            'seasonal_strength < 0.4',  # No strong seasonality
            'noise_level < 0.4'
        ],
        'priority': 'MEDIUM'
    },
    
    # Seasonal rules
    'seasonal_decomp_rule': {
        'conditions': [
            'dominant_period >= 7 and dominant_period <= 365',  # Meaningful periods
            'seasonal_strength > 0.6',
            'size >= 3 * dominant_period',  # At least 3 cycles
            'is_regular == True'  # Regular intervals required
        ],
        'priority': 'HIGH'
    },
    
    'periodic_rule': {
        'conditions': [
            'dominant_period >= 2',
            'seasonal_strength > 0.4',
            'autocorrelation_score > 0.5',
            'size >= 2 * dominant_period'
        ],
        'priority': 'MEDIUM'
    },
    
    # Contextual rules
    'consecutive_anomaly': {
        'conditions': [
            'autocorrelation_score > 0.6',  # High temporal dependence
            'anomaly_ratio_estimate > 0.03',  # Some anomalies expected
            'size >= 500'  # Need sufficient context
        ],
        'priority': 'MEDIUM'
    }
}

# MACHINE LEARNING METHODS
# ------------------------

ML_MAPPING = {
    # Isolation-based
    'isolation_forest': {
        'conditions': [
            '(n_features > 1 and size >= 2000) or '
            '(anomaly_ratio_estimate > 0.15) or '
            '(noise_level > 0.3 and size >= 1000)',
            'missing_value_ratio < 0.2'  # Can't handle too many missing values well
        ],
        'priority': 'HIGH',
        'data_requirements': 'Works well with mixed data types and high-dimensional data'
    },
    
    # Distance-based  
    'one_class_svm': {
        'conditions': [
            'n_features <= 50',  # Curse of dimensionality
            'size >= 500 and size <= 50000',  # Memory constraints
            'noise_level < 0.3',
            'missing_value_ratio < 0.05'
        ],
        'priority': 'MEDIUM'
    },
    
    'elliptic_envelope': {
        'conditions': [
            'n_features <= 20',  # Works best in lower dimensions
            'size >= 200',
            'noise_level < 0.3',
            'skew < 1.0 and kurtosis < 5',  # Assumes roughly Gaussian
            'missing_value_ratio < 0.05'
        ],
        'priority': 'MEDIUM'
    },
    
    'local_outlier_factor': {
        'conditions': [
            'n_features >= 2',  # Needs multiple dimensions for local density
            'size >= 1000 and size <= 100000',  # Computational complexity
            'anomaly_ratio_estimate < 0.20'  # Works better with fewer anomalies
        ],
        'priority': 'MEDIUM'
    },
    
    # Clustering-based
    'dbscan': {
        'conditions': [
            'n_features >= 2',
            'size >= 1000',
            'anomaly_ratio_estimate > 0.05',  # Needs some anomalies to form separate clusters
            'noise_level < 0.5',
            'has_density_based_clusters == True' 
        ],
        'priority': 'HIGH'
    },
    
    'kmeans': {
        'conditions': [
            'n_features >= 2',
            'size >= 2000',
            'anomaly_ratio_estimate > 0.05 and anomaly_ratio_estimate < 0.20',
            'skew < 1.5',  # K-means assumes spherical clusters
            'missing_value_ratio < 0.1'
        ],
        'priority': 'LOW'  # Often not the best choice for anomaly detection
    },
    
    # Statistical
    'zscore': {
        'conditions': [
            'n_features <= 10',  # Works best in lower dimensions
            'noise_level < 0.3',
            'size >= 100',
            'anomaly_ratio_estimate < 0.15',
            'skew >= -0.5 and skew <= 0.5',  # Roughly normal distribution
            'kurtosis <= 3.5'
        ],
        'priority': 'HIGH'
    },
    
    'mad': {
        'conditions': [
            'noise_level < 0.4',
            'size >= 100',
            'anomaly_ratio_estimate < 0.15',
            'skew > 0.5 or kurtosis > 3.5',  # More robust than z-score for skewed data
            'missing_value_ratio < 0.1'
        ],
        'priority': 'HIGH'
    },
    
    'ewma': {
        'conditions': [
            'n_features == 1',  # Primarily for univariate time series
            'noise_level < 0.4',
            'size >= 200',
            'autocorrelation_score > 0.3',  # Some temporal dependence needed
            'is_regular == True'
        ],
        'priority': 'MEDIUM'
    },
    
    # Time Series Models
    'prophet': {
        'conditions': [
            'n_features == 1',  # Univariate only
            'dominant_period >= 7 and dominant_period <= 365',  # Daily to yearly patterns
            'seasonal_strength > 0.6',
            'size >= 2 * 365',  # At least 2 years of daily data or equivalent
            'is_regular == True',
            'missing_value_ratio < 0.15'  # Prophet handles some missing data
        ],
        'priority': 'HIGH'
    },
    
    'arima': {
        'conditions': [
            'n_features == 1',
            'autocorrelation_score > 0.6',
            'size >= 100 and size <= 10000',  # ARIMA can be slow on large datasets
            'is_stationary == True or adf_pvalue < 0.05',
            'seasonal_strength < 0.6',  # Use SARIMA for strong seasonality
            'is_regular == True'
        ],
        'priority': 'HIGH'
    },
    
    'sarima': {
        'conditions': [
            'n_features == 1',
            'size >= 3 * dominant_period and size <= 10000',
            'is_stationary == True or adf_pvalue < 0.05',
            'autocorrelation_score > 0.6',
            'seasonal_strength > 0.8',  # Strong seasonality required
            'dominant_period >= 4 and dominant_period <= 365',
            'is_regular == True'
        ],
        'priority': 'HIGH'
    },
    
    'exponential_smoothing': {
        'conditions': [
            'n_features == 1',
            'size >= 50',
            'is_regular == True',
            '(trend_ratio > 0.3) or (seasonal_strength > 0.4)',  # Need trend or seasonality
            'missing_value_ratio < 0.1'
        ],
        'priority': 'MEDIUM'
    },
    
    # Deep Learning
    'lstm': {
        'conditions': [
            'autocorrelation_score > 0.7',  # Strong temporal patterns
            'size >= 5000',  # DL needs lots of data
            'n_features <= 50',  # Computational constraints
            'is_regular == True',
            'missing_value_ratio < 0.05'
        ],
        'priority': 'HIGH',
        'computational_cost': 'HIGH'
    },
    
    'gru': {
        'conditions': [
            'autocorrelation_score > 0.7',
            'size >= 3000',  # Slightly less data than LSTM
            'n_features <= 50',
            'is_regular == True',
            'missing_value_ratio < 0.05'
        ],
        'priority': 'HIGH',
        'computational_cost': 'HIGH'
    },
    
    'autoencoder': {
        'conditions': [
            'n_features >= 5',  # Benefits from multiple features
            'size >= 5000',
            'missing_value_ratio < 0.05',
            'noise_level > 0.2'  # Good for noisy data
        ],
        'priority': 'MEDIUM',
        'computational_cost': 'HIGH'
    },
    
    'usad': {
        'conditions': [
            'n_features > 1',  # Multivariate
            'size >= 10000',  # USAD needs lots of data
            'missing_value_ratio < 0.02',
            'autocorrelation_score > 0.5'  # Time series component
        ],
        'priority': 'MEDIUM',
        'computational_cost': 'VERY_HIGH'
    }
}


'''

# ADDITIONAL SELECTION LOGIC
# --------------------------

SELECTION_PRIORITIES = {
    'data_driven': [
        # If you have labels, prioritize methods that can use them
        'IF has_anomaly_labels == True: boost supervised methods',
        
        # Size-based priorities
        'IF size < 500: prefer simple statistical rules',
        'IF size > 50000: prefer scalable methods (Isolation Forest, simple rules)',
        
        # Quality-based priorities  
        'IF missing_value_ratio > 0.2: prefer robust methods or extensive preprocessing',
        'IF noise_level > 0.5: prefer robust methods (MAD, Isolation Forest)',
        
        # Pattern-based priorities
        'IF seasonal_strength > 0.8: strongly prefer seasonal methods',
        'IF trend_ratio > 0.6: prefer trend-aware methods',
    ],
    
    'computational_constraints': {
        'real_time': ['zscore', 'mad', 'ewma', 'static_rules'],
        'batch_processing': ['isolation_forest', 'lstm', 'prophet'],
        'limited_memory': ['statistical_rules', 'simple_ml'],
        'high_performance': ['autoencoder', 'lstm', 'usad']
    }
}

# ENSEMBLE RECOMMENDATIONS
# ------------------------

ENSEMBLE_RULES = {
    'high_confidence': {
        'conditions': 'Multiple methods agree on > 80% of predictions',
        'action': 'Use intersection of predictions'
    },
    
    'medium_confidence': {
        'conditions': 'Methods partially agree (50-80%)',
        'action': 'Use weighted voting based on method reliability'
    },
    
    'low_confidence': {
        'conditions': 'High disagreement between methods',
        'action': 'Flag for manual review or use most conservative method'
    }
}

CONDITION_WEIGHTS = {
    # General Properties - High importance for broad model class selection
    'domain_knowledge_available': 30,
    'size >= 5000': 35,
    'size >= 2000': 30,
    'size >= 1000': 25,
    'size >= 500': 20,
    'size >= 200': 15,
    'n_features > 1': 10,
    'n_features >= 2': 15,
    'n_features <= 10': 10,
    'n_features <= 20': 15,
    'n_features <= 50': 20,
    'missing_value_ratio < 0.02': 15,
    'missing_value_ratio < 0.05': 10,
    'missing_value_ratio < 0.1': 5,
    'missing_value_ratio < 0.15': 5,
    'missing_value_ratio < 0.2': 5,

    # Distributional Properties - Critical for statistical methods
    'has_negative == False': 15,
    'skew < 1.0': 20,
    'skew < 1.5': 15,
    'skew >= -0.5 and skew <= 0.5': 35,
    'skew > 0.5 or kurtosis > 3.5': 25,
    'kurtosis < 5': 20,
    'kurtosis <= 3.5': 30,
    'has_density_based_clusters == True': 50, # High impact for DBSCAN

    # Anomaly Properties - Important for method robustness
    'anomaly_ratio_estimate < 0.05': 25,
    'anomaly_ratio_estimate < 0.15': 15,
    'anomaly_ratio_estimate < 0.20': 10,
    'anomaly_ratio_estimate > 0.02': 15,
    'anomaly_ratio_estimate > 0.03': 15,
    'anomaly_ratio_estimate > 0.05': 20,
    'anomaly_ratio_estimate > 0.15': 25,

    # Temporal Properties - Crucial for time series models
    'is_regular == True': 30,
    'autocorrelation_score > 0.3': 15,
    'autocorrelation_score > 0.4': 20,
    'autocorrelation_score > 0.5': 25,
    'autocorrelation_score > 0.6': 30,
    'autocorrelation_score > 0.7': 35,
    'trend_ratio > 0.3': 20,
    'trend_ratio > 0.4': 25,
    'trend_ratio > 0.6': 30,
    'seasonal_strength > 0.4': 20,
    'seasonal_strength > 0.6': 30,
    'seasonal_strength > 0.8': 40,
    'dominant_period >= 2': 5,
    'dominant_period >= 4 and dominant_period <= 365': 20,
    'dominant_period >= 7 and dominant_period <= 365': 30,
    'size >= 2 * dominant_period': 15,
    'size >= 3 * dominant_period': 20,

    # Noise/Pattern Properties
    'noise_level < 0.3': 25,
    'noise_level < 0.4': 20,
    'noise_level < 0.5': 15,
    'roll_kurt_10 > 5 or high_pass_10 > 0.7': 30,

    # Other Specific Properties
    'is_stationary == True or adf_pvalue < 0.05': 20,
    '(n_features > 1 and size >= 2000) or (anomaly_ratio_estimate > 0.15) or (noise_level > 0.3 and size >= 1000)': 25,
}

