# Anamalitics
Step-by-Step Workflow

    📊 GENERATE Tab - Generate Data:
        Configure date range and time intervals
        Set peak/off hours for your use case
        Adjust seasonality and trend parameters
        Configure anomaly types and rates
        Click "Generate Synthetic Data"
        Visualize the Generated Data w.r.t different categories of anomalies in that data
        For Manual editing of any data sample, please refer the "Manual Editing" section.
        Export the data or config for future usage
    🔍 ANALYZE Tab - Run Anomaly Detection:
        Choose to use generated data or upload your own CSV
        Select & Configure rule-based methods
        Select & Configure ML models (configure ensemble & Threshold , for multiple ML Models usage)
        Click "Run Anomaly Detection"
        Detection Results : contains the comparision between anomaly detection by Rule-Based and ML-Based selected methods
    📈 COMPARE Tab - Analyze Results:
        View performance metrics for all methods
        Compare precision, recall, F1-scores
        Analyze method agreements
        View the performance of individual Rule-Methods and ML-Models w.r.t different anomaly categories and time
        Export results for further analysis
    ✏️ Manual Editing (Optional):
        In GENERATE tab, use point-by-point editing
        Select points by index, timestamp, or value range
        Mark points as anomalies or normal
        Apply bulk operations

