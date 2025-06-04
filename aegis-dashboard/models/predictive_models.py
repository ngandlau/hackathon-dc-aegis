"""
Predictive Models and Analytics Engine
=====================================

Core machine learning and analytics components for disease outbreak prediction:
- Time series forecasting models
- Social media sentiment analysis
- Ensemble prediction methods
- Anomaly detection
- Early warning signal generation

Date: June 3, 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """
    Advanced time series forecasting for hospitalization data
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'new_covid_19_hospital_admissions'
        self.sequence_length = 14  # Use 14 days of history
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features from raw data"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['jurisdiction', 'date'])
        
        # Create time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Create lag features for each jurisdiction
        for jurisdiction in df['jurisdiction'].unique():
            mask = df['jurisdiction'] == jurisdiction
            jurisdiction_data = df[mask].copy()
            
            # Lag features (1, 3, 7, 14 days)
            for lag in [1, 3, 7, 14]:
                lag_col = f'{self.target_column}_lag_{lag}'
                df.loc[mask, lag_col] = jurisdiction_data[self.target_column].shift(lag)
            
            # Rolling statistics (7, 14 days)
            for window in [7, 14]:
                rolling_mean_col = f'{self.target_column}_rolling_mean_{window}'
                rolling_std_col = f'{self.target_column}_rolling_std_{window}'
                
                df.loc[mask, rolling_mean_col] = jurisdiction_data[self.target_column].rolling(window=window).mean()
                df.loc[mask, rolling_std_col] = jurisdiction_data[self.target_column].rolling(window=window).std()
            
            # Trend features
            df.loc[mask, 'trend_7d'] = jurisdiction_data[self.target_column].rolling(window=7).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
            )
        
        # Seasonal decomposition features
        df['seasonal_component'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['seasonal_component_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, jurisdiction: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        jurisdiction_data = df[df['jurisdiction'] == jurisdiction].copy()
        jurisdiction_data = jurisdiction_data.sort_values('date')
        
        # Select feature columns (exclude non-numeric and target)
        exclude_cols = ['date', 'jurisdiction', self.target_column]
        feature_cols = [col for col in jurisdiction_data.columns 
                       if col not in exclude_cols and jurisdiction_data[col].dtype in ['int64', 'float64']]
        
        # Remove rows with NaN values
        jurisdiction_data = jurisdiction_data.dropna()
        
        if len(jurisdiction_data) < self.sequence_length + 1:
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(self.sequence_length, len(jurisdiction_data)):
            # Use sequence of features
            sequence = jurisdiction_data[feature_cols].iloc[i-self.sequence_length:i].values
            target = jurisdiction_data[self.target_column].iloc[i]
            
            X.append(sequence.flatten())  # Flatten the sequence
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the forecasting models"""
        logger.info("Training time series forecasting models...")
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Combine data from all jurisdictions
        all_X, all_y = [], []
        
        for jurisdiction in df_features['jurisdiction'].unique():
            X_seq, y_seq = self.create_sequences(df_features, jurisdiction)
            if len(X_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
        
        if not all_X:
            raise ValueError("No valid sequences created for training")
        
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Store feature dimension for later use
        self.feature_dim = X.shape[1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Scale targets
        self.scalers['target'] = StandardScaler()
        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train multiple models
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }
        
        results = {}
        for name, model in models_config.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = self.scalers['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'mae': mae, 'mse': mse, 'r2': r2}
            self.models[name] = model
            
            logger.info(f"{name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.3f}")
        
        # Create ensemble model
        self._create_ensemble_model(X_test_scaled, y_test)
        
        return results
    
    def _create_ensemble_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Create ensemble model with weighted predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            y_pred_scaled = model.predict(X_test)
            y_pred = self.scalers['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            predictions[name] = y_pred
        
        # Calculate weights based on inverse MAE
        weights = {}
        total_weight = 0
        
        for name, y_pred in predictions.items():
            mae = mean_absolute_error(y_test, y_pred)
            weight = 1 / (mae + 1e-6)  # Add small epsilon to avoid division by zero
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        self.ensemble_weights = weights
        logger.info(f"Ensemble weights: {weights}")
    
    def predict(self, df: pd.DataFrame, jurisdiction: str, days_ahead: int = 7) -> Dict:
        """Make predictions for a specific jurisdiction"""
        df_features = self.prepare_features(df)
        jurisdiction_data = df_features[df_features['jurisdiction'] == jurisdiction].copy()
        jurisdiction_data = jurisdiction_data.sort_values('date').dropna()
        
        if len(jurisdiction_data) < self.sequence_length:
            return {'error': 'Insufficient data for prediction'}
        
        # Get the most recent sequence
        exclude_cols = ['date', 'jurisdiction', self.target_column]
        feature_cols = [col for col in jurisdiction_data.columns 
                       if col not in exclude_cols and jurisdiction_data[col].dtype in ['int64', 'float64']]
        
        recent_sequence = jurisdiction_data[feature_cols].tail(self.sequence_length).values.flatten()
        
        if len(recent_sequence) != self.feature_dim:
            return {'error': 'Feature dimension mismatch'}
        
        # Scale features
        recent_sequence_scaled = self.scalers['features'].transform(recent_sequence.reshape(1, -1))
        
        # Make predictions with all models
        predictions = {}
        for name, model in self.models.items():
            pred_scaled = model.predict(recent_sequence_scaled)
            pred = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions[name] = max(0, pred)  # Ensure non-negative predictions
        
        # Ensemble prediction
        ensemble_pred = sum(predictions[name] * self.ensemble_weights[name] 
                          for name in predictions)
        
        # Generate forecast for multiple days
        forecast = []
        current_date = jurisdiction_data['date'].max()
        
        for day in range(1, days_ahead + 1):
            forecast_date = current_date + timedelta(days=day)
            # For simplicity, use the same prediction (in practice, would use recursive forecasting)
            forecast.append({
                'date': forecast_date,
                'predicted_admissions': ensemble_pred,
                'confidence_lower': ensemble_pred * 0.8,
                'confidence_upper': ensemble_pred * 1.2
            })
        
        return {
            'jurisdiction': jurisdiction,
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'forecast': forecast,
            'model_weights': self.ensemble_weights
        }

class SocialMediaAnalyzer:
    """
    Analyzes social media data for early warning signals
    """
    
    def __init__(self):
        self.sentiment_weights = {'negative': -1, 'neutral': 0, 'positive': 1}
        self.symptom_keywords = [
            'fever', 'cough', 'fatigue', 'headache', 'sore throat',
            'loss of taste', 'loss of smell', 'shortness of breath', 'body aches'
        ]
        
    def analyze_sentiment_trends(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment trends over time"""
        if 'sentiment' not in social_data.columns:
            return pd.DataFrame()
        
        social_data['date'] = pd.to_datetime(social_data['date'])
        
        # Calculate daily sentiment scores
        daily_sentiment = social_data.groupby('date').agg({
            'sentiment': lambda x: sum(self.sentiment_weights.get(s, 0) for s in x) / len(x)
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        # Calculate rolling averages
        daily_sentiment['sentiment_7d'] = daily_sentiment['avg_sentiment'].rolling(window=7).mean()
        daily_sentiment['sentiment_14d'] = daily_sentiment['avg_sentiment'].rolling(window=14).mean()
        
        return daily_sentiment
    
    def extract_symptom_mentions(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Extract and count symptom mentions over time"""
        if 'content' not in social_data.columns and 'title' not in social_data.columns:
            return pd.DataFrame()
        
        social_data['date'] = pd.to_datetime(social_data['date'])
        
        # Combine content fields
        if 'content' in social_data.columns:
            text_field = 'content'
        else:
            text_field = 'title'
        
        # Count symptom mentions
        symptom_data = []
        
        for _, row in social_data.iterrows():
            text = str(row[text_field]).lower()
            for symptom in self.symptom_keywords:
                if symptom in text:
                    symptom_data.append({
                        'date': row['date'],
                        'symptom': symptom,
                        'mentions': 1
                    })
        
        if not symptom_data:
            return pd.DataFrame()
        
        symptom_df = pd.DataFrame(symptom_data)
        
        # Aggregate by date and symptom
        daily_symptoms = symptom_df.groupby(['date', 'symptom']).agg({
            'mentions': 'sum'
        }).reset_index()
        
        # Calculate total daily symptom mentions
        total_daily = daily_symptoms.groupby('date').agg({
            'mentions': 'sum'
        }).reset_index()
        total_daily.columns = ['date', 'total_symptom_mentions']
        
        return total_daily
    
    def calculate_health_concern_index(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate a composite health concern index from social media"""
        social_data['date'] = pd.to_datetime(social_data['date'])
        
        # Get sentiment trends
        sentiment_trends = self.analyze_sentiment_trends(social_data)
        
        # Get symptom mentions
        symptom_mentions = self.extract_symptom_mentions(social_data)
        
        # Calculate engagement metrics
        engagement_cols = ['engagement_score', 'retweet_count', 'like_count', 'upvotes', 'comments']
        available_engagement = [col for col in engagement_cols if col in social_data.columns]
        
        if available_engagement:
            daily_engagement = social_data.groupby('date')[available_engagement].mean().reset_index()
            daily_engagement['avg_engagement'] = daily_engagement[available_engagement].mean(axis=1)
        else:
            daily_engagement = pd.DataFrame()
        
        # Combine all metrics
        concern_index = sentiment_trends.copy() if not sentiment_trends.empty else pd.DataFrame()
        
        if not symptom_mentions.empty:
            concern_index = concern_index.merge(symptom_mentions, on='date', how='outer')
        
        if not daily_engagement.empty:
            concern_index = concern_index.merge(daily_engagement[['date', 'avg_engagement']], on='date', how='outer')
        
        if concern_index.empty:
            return pd.DataFrame()
        
        # Fill missing values
        concern_index = concern_index.fillna(0)
        
        # Normalize metrics to 0-1 scale
        scaler = MinMaxScaler()
        numeric_cols = concern_index.select_dtypes(include=[np.number]).columns
        concern_index[numeric_cols] = scaler.fit_transform(concern_index[numeric_cols])
        
        # Calculate composite concern index
        weights = {
            'avg_sentiment': -0.3,  # Negative sentiment increases concern
            'total_symptom_mentions': 0.4,  # More symptom mentions increase concern
            'avg_engagement': 0.3  # Higher engagement increases concern
        }
        
        concern_index['health_concern_index'] = 0
        for metric, weight in weights.items():
            if metric in concern_index.columns:
                concern_index['health_concern_index'] += concern_index[metric] * weight
        
        # Normalize to 0-100 scale
        concern_index['health_concern_index'] = (concern_index['health_concern_index'] + 1) * 50
        
        return concern_index[['date', 'health_concern_index']]

class AnomalyDetector:
    """
    Detects anomalies in health data that may indicate emerging outbreaks
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        
    def detect_hospitalization_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in hospitalization data"""
        from sklearn.ensemble import IsolationForest
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        anomalies = []
        
        for jurisdiction in df['jurisdiction'].unique():
            jurisdiction_data = df[df['jurisdiction'] == jurisdiction].copy()
            jurisdiction_data = jurisdiction_data.sort_values('date')
            
            if len(jurisdiction_data) < 10:  # Need minimum data points
                continue
            
            # Prepare features for anomaly detection
            features = ['new_covid_19_hospital_admissions']
            if 'covid_19_inpatient_bed_occupancy_7_day_average' in jurisdiction_data.columns:
                features.append('covid_19_inpatient_bed_occupancy_7_day_average')
            
            # Remove missing values
            feature_data = jurisdiction_data[features].dropna()
            
            if len(feature_data) < 5:
                continue
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(feature_data)
            
            # Add anomaly information
            feature_indices = feature_data.index
            for i, (idx, label) in enumerate(zip(feature_indices, anomaly_labels)):
                anomalies.append({
                    'date': jurisdiction_data.loc[idx, 'date'],
                    'jurisdiction': jurisdiction,
                    'is_anomaly': label == -1,
                    'anomaly_score': iso_forest.decision_function(feature_data.iloc[[i]])[0],
                    'admissions': jurisdiction_data.loc[idx, 'new_covid_19_hospital_admissions']
                })
        
        return pd.DataFrame(anomalies)
    
    def detect_social_media_anomalies(self, social_concern_index: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in social media concern patterns"""
        from sklearn.ensemble import IsolationForest
        
        if social_concern_index.empty or 'health_concern_index' not in social_concern_index.columns:
            return pd.DataFrame()
        
        # Prepare data
        concern_data = social_concern_index[['health_concern_index']].dropna()
        
        if len(concern_data) < 10:
            return pd.DataFrame()
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(concern_data)
        
        # Create results
        anomalies = []
        for i, (idx, label) in enumerate(zip(concern_data.index, anomaly_labels)):
            anomalies.append({
                'date': social_concern_index.loc[idx, 'date'],
                'is_anomaly': label == -1,
                'anomaly_score': iso_forest.decision_function(concern_data.iloc[[i]])[0],
                'concern_index': social_concern_index.loc[idx, 'health_concern_index']
            })
        
        return pd.DataFrame(anomalies)

class EarlyWarningSystem:
    """
    Integrates all models to generate early warning alerts
    """
    
    def __init__(self):
        self.forecaster = TimeSeriesForecaster()
        self.social_analyzer = SocialMediaAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.alert_thresholds = {
            'hospitalization_increase': 0.25,  # 25% increase
            'concern_index_threshold': 70,     # High concern level
            'anomaly_threshold': -0.5          # Anomaly score threshold
        }
        
    def generate_alerts(self, 
                       cdc_data: pd.DataFrame,
                       social_data: pd.DataFrame,
                       jurisdiction: str = None) -> Dict:
        """Generate comprehensive early warning alerts"""
        
        alerts = {
            'timestamp': datetime.now(),
            'jurisdiction': jurisdiction,
            'alert_level': 'normal',  # normal, elevated, high, critical
            'alerts': [],
            'predictions': {},
            'anomalies': {},
            'recommendations': []
        }
        
        try:
            # Analyze social media trends
            if not social_data.empty:
                concern_index = self.social_analyzer.calculate_health_concern_index(social_data)
                
                if not concern_index.empty:
                    latest_concern = concern_index['health_concern_index'].iloc[-1]
                    alerts['predictions']['health_concern_index'] = latest_concern
                    
                    if latest_concern > self.alert_thresholds['concern_index_threshold']:
                        alerts['alerts'].append({
                            'type': 'social_media_concern',
                            'severity': 'high',
                            'message': f'Health concern index elevated: {latest_concern:.1f}',
                            'data_source': 'social_media'
                        })
                
                # Detect social media anomalies
                social_anomalies = self.anomaly_detector.detect_social_media_anomalies(concern_index)
                if not social_anomalies.empty:
                    recent_anomalies = social_anomalies[social_anomalies['is_anomaly'] == True]
                    if len(recent_anomalies) > 0:
                        alerts['anomalies']['social_media'] = len(recent_anomalies)
                        alerts['alerts'].append({
                            'type': 'social_media_anomaly',
                            'severity': 'medium',
                            'message': f'{len(recent_anomalies)} social media anomalies detected',
                            'data_source': 'social_media'
                        })
            
            # Analyze hospitalization data
            if not cdc_data.empty:
                # Detect hospitalization anomalies
                hosp_anomalies = self.anomaly_detector.detect_hospitalization_anomalies(cdc_data)
                
                if not hosp_anomalies.empty:
                    if jurisdiction:
                        jurisdiction_anomalies = hosp_anomalies[
                            (hosp_anomalies['jurisdiction'] == jurisdiction) & 
                            (hosp_anomalies['is_anomaly'] == True)
                        ]
                    else:
                        jurisdiction_anomalies = hosp_anomalies[hosp_anomalies['is_anomaly'] == True]
                    
                    if len(jurisdiction_anomalies) > 0:
                        alerts['anomalies']['hospitalization'] = len(jurisdiction_anomalies)
                        alerts['alerts'].append({
                            'type': 'hospitalization_anomaly',
                            'severity': 'high',
                            'message': f'{len(jurisdiction_anomalies)} hospitalization anomalies detected',
                            'data_source': 'cdc'
                        })
                
                # Make predictions if we have enough data
                if jurisdiction and len(cdc_data[cdc_data['jurisdiction'] == jurisdiction]) > 14:
                    try:
                        # Train forecaster if not already trained
                        if not self.forecaster.models:
                            self.forecaster.train(cdc_data)
                        
                        prediction = self.forecaster.predict(cdc_data, jurisdiction)
                        if 'error' not in prediction:
                            alerts['predictions']['hospitalization_forecast'] = prediction
                            
                            # Check for predicted increase
                            current_avg = cdc_data[cdc_data['jurisdiction'] == jurisdiction][
                                'new_covid_19_hospital_admissions'
                            ].tail(7).mean()
                            
                            predicted_avg = prediction['ensemble_prediction']
                            
                            if predicted_avg > current_avg * (1 + self.alert_thresholds['hospitalization_increase']):
                                alerts['alerts'].append({
                                    'type': 'predicted_increase',
                                    'severity': 'high',
                                    'message': f'Predicted {((predicted_avg/current_avg - 1) * 100):.1f}% increase in hospitalizations',
                                    'data_source': 'prediction'
                                })
                    except Exception as e:
                        logger.warning(f"Prediction failed: {e}")
            
            # Determine overall alert level
            alert_levels = [alert['severity'] for alert in alerts['alerts']]
            if 'critical' in alert_levels:
                alerts['alert_level'] = 'critical'
            elif 'high' in alert_levels:
                alerts['alert_level'] = 'high'
            elif 'medium' in alert_levels:
                alerts['alert_level'] = 'elevated'
            
            # Generate recommendations
            alerts['recommendations'] = self._generate_recommendations(alerts)
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            alerts['error'] = str(e)
        
        return alerts
    
    def _generate_recommendations(self, alerts: Dict) -> List[str]:
        """Generate actionable recommendations based on alerts"""
        recommendations = []
        
        alert_level = alerts['alert_level']
        alert_types = [alert['type'] for alert in alerts['alerts']]
        
        if alert_level in ['high', 'critical']:
            recommendations.append("Increase surveillance and monitoring activities")
            recommendations.append("Review and update response plans")
            
        if 'social_media_concern' in alert_types:
            recommendations.append("Monitor public communication channels for misinformation")
            recommendations.append("Prepare public health messaging to address concerns")
            
        if 'hospitalization_anomaly' in alert_types:
            recommendations.append("Investigate unusual hospitalization patterns")
            recommendations.append("Assess hospital capacity and resource needs")
            
        if 'predicted_increase' in alert_types:
            recommendations.append("Prepare for potential surge in healthcare demand")
            recommendations.append("Consider implementing preventive measures")
            
        if not recommendations:
            recommendations.append("Continue routine monitoring and surveillance")
            
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # This will be used for testing the models
    print("Predictive Models and Analytics Engine initialized")
    print("Ready for integration with data sources and dashboards")

