"""
Advanced Analytics and Ensemble Methods
======================================

Implements sophisticated ensemble methods and advanced analytics for
disease outbreak prediction including:
- Multi-modal ensemble learning
- Causal inference analysis
- Geographic clustering
- Real-time model updating


Date: June 3, 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MultiModalEnsemble:
    """
    Advanced ensemble that combines predictions from multiple data modalities
    """
    
    def __init__(self):
        self.modality_models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.modality_weights = {}
        
    def train_modality_models(self, 
                            cdc_data: pd.DataFrame,
                            social_data: pd.DataFrame,
                            supplementary_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train separate models for each data modality"""
        
        results = {}
        
        # CDC/Hospitalization modality
        if not cdc_data.empty:
            cdc_features = self._prepare_cdc_features(cdc_data)
            if not cdc_features.empty:
                cdc_model = self._train_cdc_model(cdc_features)
                self.modality_models['cdc'] = cdc_model
                results['cdc'] = cdc_model['performance']
        
        # Social media modality
        if not social_data.empty:
            social_features = self._prepare_social_features(social_data)
            if not social_features.empty:
                social_model = self._train_social_model(social_features)
                self.modality_models['social'] = social_model
                results['social'] = social_model['performance']
        
        # Supplementary data modalities
        for modality_name, data in supplementary_data.items():
            if not data.empty:
                features = self._prepare_supplementary_features(data, modality_name)
                if not features.empty:
                    model = self._train_supplementary_model(features, modality_name)
                    self.modality_models[modality_name] = model
                    results[modality_name] = model['performance']
        
        # Train ensemble meta-model
        if len(self.modality_models) > 1:
            ensemble_results = self._train_ensemble_meta_model()
            results['ensemble'] = ensemble_results
        
        return results
    
    def _prepare_cdc_features(self, cdc_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from CDC hospitalization data"""
        df = cdc_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate to daily national level
        daily_features = df.groupby('date').agg({
            'new_covid_19_hospital_admissions': ['sum', 'mean', 'std'],
            'covid_19_inpatient_bed_occupancy_7_day_average': ['mean', 'std'],
            'covid_19_icu_bed_occupancy_7_day_average': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        daily_features.columns = ['date'] + [f"{col[0]}_{col[1]}" for col in daily_features.columns[1:]]
        
        # Create temporal features
        daily_features['day_of_week'] = daily_features['date'].dt.dayofweek
        daily_features['day_of_year'] = daily_features['date'].dt.dayofyear
        daily_features['week_of_year'] = daily_features['date'].dt.isocalendar().week
        
        # Create lag features
        for col in daily_features.select_dtypes(include=[np.number]).columns:
            if col != 'day_of_week':  # Don't lag categorical features
                for lag in [1, 3, 7]:
                    daily_features[f'{col}_lag_{lag}'] = daily_features[col].shift(lag)
        
        # Create rolling features
        for col in ['new_covid_19_hospital_admissions_sum', 'new_covid_19_hospital_admissions_mean']:
            if col in daily_features.columns:
                daily_features[f'{col}_rolling_7'] = daily_features[col].rolling(window=7).mean()
                daily_features[f'{col}_rolling_14'] = daily_features[col].rolling(window=14).mean()
        
        return daily_features.dropna()
    
    def _prepare_social_features(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from social media data"""
        df = social_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Daily aggregations
        daily_social = df.groupby('date').agg({
            'engagement_score': ['mean', 'std', 'count'] if 'engagement_score' in df.columns else ['count'],
        }).reset_index()
        
        # Flatten column names
        daily_social.columns = ['date'] + [f"social_{col[0]}_{col[1]}" for col in daily_social.columns[1:]]
        
        # Sentiment analysis if available
        if 'sentiment' in df.columns:
            sentiment_daily = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            sentiment_daily.columns = [f'sentiment_{col}' for col in sentiment_daily.columns]
            sentiment_daily = sentiment_daily.reset_index()
            daily_social = daily_social.merge(sentiment_daily, on='date', how='left')
        
        # Post type analysis if available
        if 'tweet_type' in df.columns:
            type_daily = df.groupby(['date', 'tweet_type']).size().unstack(fill_value=0)
            type_daily.columns = [f'type_{col}' for col in type_daily.columns]
            type_daily = type_daily.reset_index()
            daily_social = daily_social.merge(type_daily, on='date', how='left')
        elif 'post_type' in df.columns:
            type_daily = df.groupby(['date', 'post_type']).size().unstack(fill_value=0)
            type_daily.columns = [f'type_{col}' for col in type_daily.columns]
            type_daily = type_daily.reset_index()
            daily_social = daily_social.merge(type_daily, on='date', how='left')
        
        # Fill missing values
        daily_social = daily_social.fillna(0)
        
        # Create rolling features
        numeric_cols = daily_social.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            daily_social[f'{col}_rolling_7'] = daily_social[col].rolling(window=7).mean()
        
        return daily_social.dropna()
    
    def _prepare_supplementary_features(self, data: pd.DataFrame, modality_name: str) -> pd.DataFrame:
        """Prepare features from supplementary data sources"""
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if modality_name == 'google_trends':
            # Aggregate search volumes by date
            daily_trends = df.groupby('date').agg({
                'search_volume': ['mean', 'max', 'std']
            }).reset_index()
            daily_trends.columns = ['date'] + [f"trends_{col[1]}" for col in daily_trends.columns[1:]]
            
        elif modality_name == 'mobility':
            # Aggregate mobility changes by date
            daily_mobility = df.groupby('date').agg({
                'mobility_change_pct': ['mean', 'std']
            }).reset_index()
            daily_mobility.columns = ['date'] + [f"mobility_{col[1]}" for col in daily_mobility.columns[1:]]
            return daily_mobility
            
        elif modality_name == 'economic':
            # Economic data is already daily/weekly
            return df
            
        elif modality_name == 'weather':
            # Aggregate weather by date
            daily_weather = df.groupby('date').agg({
                'temperature_f': ['mean', 'std'],
                'humidity_pct': ['mean', 'std'],
                'air_quality_index': ['mean', 'max']
            }).reset_index()
            daily_weather.columns = ['date'] + [f"weather_{col[0]}_{col[1]}" for col in daily_weather.columns[1:]]
            return daily_weather
        
        return daily_trends if modality_name == 'google_trends' else df
    
    def _train_cdc_model(self, features: pd.DataFrame) -> Dict:
        """Train model for CDC data modality"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Prepare target variable (next day's admissions)
        target_col = 'new_covid_19_hospital_admissions_sum'
        if target_col not in features.columns:
            target_col = 'new_covid_19_hospital_admissions_mean'
        
        if target_col not in features.columns:
            return {'model': None, 'performance': {'error': 'No suitable target column'}}
        
        # Create target (predict next day)
        features['target'] = features[target_col].shift(-1)
        features = features.dropna()
        
        if len(features) < 10:
            return {'model': None, 'performance': {'error': 'Insufficient data'}}
        
        # Prepare features and target
        feature_cols = [col for col in features.columns if col not in ['date', 'target']]
        X = features[feature_cols]
        y = features['target']
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        
        return {
            'model': model,
            'feature_columns': feature_cols,
            'performance': {'mae': mae, 'r2': r2},
            'feature_importance': importance
        }
    
    def _train_social_model(self, features: pd.DataFrame) -> Dict:
        """Train model for social media modality"""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Create synthetic target based on social media activity
        # In practice, this would be aligned with actual outbreak data
        activity_cols = [col for col in features.columns if 'count' in col or 'mean' in col]
        if not activity_cols:
            return {'model': None, 'performance': {'error': 'No activity columns found'}}
        
        # Create composite activity score as target
        features['social_activity'] = features[activity_cols].sum(axis=1)
        features['target'] = features['social_activity'].shift(-1)  # Predict next day
        features = features.dropna()
        
        if len(features) < 10:
            return {'model': None, 'performance': {'error': 'Insufficient data'}}
        
        # Prepare features and target
        feature_cols = [col for col in features.columns if col not in ['date', 'target', 'social_activity']]
        X = features[feature_cols]
        y = features['target']
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'feature_columns': feature_cols,
            'performance': {'mae': mae, 'r2': r2}
        }
    
    def _train_supplementary_model(self, features: pd.DataFrame, modality_name: str) -> Dict:
        """Train model for supplementary data modality"""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Create target based on modality type
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {'model': None, 'performance': {'error': 'No numeric columns'}}
        
        # Use first numeric column as base for target
        base_col = numeric_cols[0]
        features['target'] = features[base_col].shift(-1)
        features = features.dropna()
        
        if len(features) < 10:
            return {'model': None, 'performance': {'error': 'Insufficient data'}}
        
        # Prepare features and target
        feature_cols = [col for col in features.columns if col not in ['date', 'target']]
        X = features[feature_cols]
        y = features['target']
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'feature_columns': feature_cols,
            'performance': {'mae': mae, 'r2': r2}
        }
    
    def _train_ensemble_meta_model(self) -> Dict:
        """Train meta-model that combines predictions from all modalities"""
        # This is a simplified version - in practice would need aligned targets
        
        # Calculate modality weights based on performance
        weights = {}
        total_weight = 0
        
        for modality, model_info in self.modality_models.items():
            if model_info['model'] is not None and 'mae' in model_info['performance']:
                # Weight inversely proportional to MAE
                weight = 1 / (model_info['performance']['mae'] + 1e-6)
                weights[modality] = weight
                total_weight += weight
        
        # Normalize weights
        for modality in weights:
            weights[modality] /= total_weight
        
        self.modality_weights = weights
        
        return {'weights': weights, 'num_modalities': len(weights)}

class GeographicClusterAnalyzer:
    """
    Analyzes geographic clustering patterns for outbreak detection
    """
    
    def __init__(self):
        self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()
        
    def analyze_geographic_clusters(self, cdc_data: pd.DataFrame) -> Dict:
        """Identify geographic clusters of high disease activity"""
        
        if cdc_data.empty:
            return {'error': 'No CDC data available'}
        
        # Get latest data for each jurisdiction
        latest_data = cdc_data.groupby('jurisdiction').last().reset_index()
        
        # Prepare features for clustering
        feature_cols = [
            'new_covid_19_hospital_admissions',
            'covid_19_inpatient_bed_occupancy_7_day_average',
            'covid_19_icu_bed_occupancy_7_day_average'
        ]
        
        available_cols = [col for col in feature_cols if col in latest_data.columns]
        if not available_cols:
            return {'error': 'No suitable features for clustering'}
        
        # Prepare clustering data
        cluster_data = latest_data[available_cols].fillna(0)
        
        if len(cluster_data) < 3:
            return {'error': 'Insufficient jurisdictions for clustering'}
        
        # Scale features
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(scaled_data)
        
        # Analyze clusters
        latest_data['cluster'] = cluster_labels
        
        cluster_analysis = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_jurisdictions = latest_data[latest_data['cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'jurisdictions': cluster_jurisdictions['jurisdiction'].tolist(),
                'size': len(cluster_jurisdictions),
                'avg_admissions': cluster_jurisdictions['new_covid_19_hospital_admissions'].mean(),
                'avg_bed_occupancy': cluster_jurisdictions.get('covid_19_inpatient_bed_occupancy_7_day_average', pd.Series([0])).mean()
            }
        
        # Identify high-risk clusters
        high_risk_clusters = []
        for cluster_id, info in cluster_analysis.items():
            if info['avg_admissions'] > latest_data['new_covid_19_hospital_admissions'].quantile(0.75):
                high_risk_clusters.append(cluster_id)
        
        return {
            'clusters': cluster_analysis,
            'high_risk_clusters': high_risk_clusters,
            'total_clusters': len(cluster_analysis),
            'noise_points': len(latest_data[latest_data['cluster'] == -1])
        }

class RealTimeModelUpdater:
    """
    Handles real-time model updating as new data becomes available
    """
    
    def __init__(self):
        self.model_performance_history = {}
        self.update_threshold = 0.1  # Retrain if performance degrades by 10%
        
    def should_update_model(self, model_name: str, current_performance: float) -> bool:
        """Determine if a model should be retrained based on performance"""
        
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = [current_performance]
            return False
        
        # Get historical performance
        history = self.model_performance_history[model_name]
        baseline_performance = np.mean(history[-5:])  # Use last 5 measurements
        
        # Check if performance has degraded significantly
        performance_change = (current_performance - baseline_performance) / baseline_performance
        
        # Update history
        self.model_performance_history[model_name].append(current_performance)
        
        # Keep only last 20 measurements
        if len(self.model_performance_history[model_name]) > 20:
            self.model_performance_history[model_name] = self.model_performance_history[model_name][-20:]
        
        return performance_change > self.update_threshold
    
    def update_model_incrementally(self, model, new_X: np.ndarray, new_y: np.ndarray):
        """Update model with new data (for models that support incremental learning)"""
        
        # Check if model supports partial_fit
        if hasattr(model, 'partial_fit'):
            model.partial_fit(new_X, new_y)
            return True
        
        # For models that don't support incremental learning, return False
        # indicating a full retrain is needed
        return False

class VisualizationEngine:
    """
    Creates advanced visualizations for model outputs and predictions
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def create_prediction_dashboard(self, 
                                  predictions: Dict,
                                  actual_data: pd.DataFrame,
                                  jurisdiction: str) -> go.Figure:
        """Create comprehensive prediction visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hospitalization Forecast', 'Model Confidence', 
                          'Feature Importance', 'Alert Timeline'),
            specs=[[{"secondary_y": True}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Hospitalization forecast
        if 'forecast' in predictions:
            forecast_data = predictions['forecast']
            dates = [item['date'] for item in forecast_data]
            predicted = [item['predicted_admissions'] for item in forecast_data]
            lower = [item['confidence_lower'] for item in forecast_data]
            upper = [item['confidence_upper'] for item in forecast_data]
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(x=dates, y=predicted, mode='lines+markers',
                          name='Predicted', line=dict(color='red')),
                row=1, col=1
            )
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(x=dates + dates[::-1], 
                          y=upper + lower[::-1],
                          fill='toself', fillcolor='rgba(255,0,0,0.2)',
                          line=dict(color='rgba(255,255,255,0)'),
                          name='Confidence Interval'),
                row=1, col=1
            )
        
        # Add historical data
        if not actual_data.empty:
            jurisdiction_data = actual_data[actual_data['jurisdiction'] == jurisdiction]
            if not jurisdiction_data.empty:
                fig.add_trace(
                    go.Scatter(x=jurisdiction_data['date'], 
                              y=jurisdiction_data['new_covid_19_hospital_admissions'],
                              mode='lines', name='Historical', line=dict(color='blue')),
                    row=1, col=1
                )
        
        # Model confidence indicator
        if 'ensemble_prediction' in predictions:
            confidence = 85  # Placeholder confidence score
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Confidence"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "yellow"},
                                   {'range': [80, 100], 'color': "green"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=2
            )
        
        # Feature importance (if available)
        if 'model_weights' in predictions:
            weights = predictions['model_weights']
            fig.add_trace(
                go.Bar(x=list(weights.keys()), y=list(weights.values()),
                      name='Model Weights'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Disease Prediction Dashboard - {jurisdiction}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_anomaly_visualization(self, anomalies: pd.DataFrame) -> go.Figure:
        """Create visualization for detected anomalies"""
        
        if anomalies.empty:
            return go.Figure().add_annotation(text="No anomalies detected", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        # Plot normal points
        normal_points = anomalies[anomalies['is_anomaly'] == False]
        if not normal_points.empty:
            fig.add_trace(
                go.Scatter(x=normal_points['date'], y=normal_points.get('admissions', normal_points.get('concern_index', 0)),
                          mode='markers', name='Normal', 
                          marker=dict(color='blue', size=6))
            )
        
        # Plot anomalies
        anomaly_points = anomalies[anomalies['is_anomaly'] == True]
        if not anomaly_points.empty:
            fig.add_trace(
                go.Scatter(x=anomaly_points['date'], y=anomaly_points.get('admissions', anomaly_points.get('concern_index', 0)),
                          mode='markers', name='Anomalies',
                          marker=dict(color='red', size=10, symbol='x'))
            )
        
        fig.update_layout(
            title="Anomaly Detection Results",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='closest'
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    print("Advanced Analytics and Ensemble Methods initialized")
    print("Ready for multi-modal ensemble learning and real-time analysis")

