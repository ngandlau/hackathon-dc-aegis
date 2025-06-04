"""
Configuration Management for Disease Early Warning System
=========================================================

Centralized configuration for all system components including:
- Data source configurations
- API endpoints and credentials
- Model parameters
- Dashboard settings
- Stakeholder-specific configurations

Date: June 3, 2025
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path

@dataclass
class CDCConfig:
    """Configuration for CDC data sources"""
    base_url: str = "https://data.cdc.gov/resource"
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        'hospitalization': '39z2-9zu6.json',
        'case_surveillance': 'vbim-akqf.json', 
        'weekly_metrics': 'akn2-qxic.json'
    })
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class SocialMediaConfig:
    """Configuration for social media data collection"""
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "Disease-Early-Warning-System/1.0"
    
    # Simulation parameters
    use_simulation: bool = True
    simulation_seed: int = 42
    
    # Keywords for data collection
    health_keywords: List[str] = field(default_factory=lambda: [
        'covid', 'coronavirus', 'symptoms', 'fever', 'cough', 'hospital',
        'sick', 'illness', 'outbreak', 'pandemic', 'quarantine', 'isolation'
    ])

@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    # Time series forecasting
    prophet_params: Dict = field(default_factory=lambda: {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative'
    })
    
    # LSTM parameters
    lstm_params: Dict = field(default_factory=lambda: {
        'sequence_length': 14,
        'hidden_units': 50,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 32
    })
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'prophet': 0.3,
        'lstm': 0.25,
        'arima': 0.2,
        'social_sentiment': 0.15,
        'mobility': 0.1
    })
    
    # Model validation
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    backtesting_window: int = 30  # days

@dataclass
class DashboardConfig:
    """Configuration for dashboard applications"""
    # Streamlit settings
    page_title: str = "Disease Early Warning System"
    page_icon: str = "🏥"
    layout: str = "wide"
    
    # Refresh intervals (seconds)
    data_refresh_interval: int = 300  # 5 minutes
    model_refresh_interval: int = 3600  # 1 hour
    
    # Visualization settings
    default_theme: str = "plotly_white"
    color_palette: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # Map settings
    default_map_center: Dict[str, float] = field(default_factory=lambda: {
        'lat': 39.8283, 'lon': -98.5795  # Geographic center of US
    })
    default_map_zoom: int = 4

@dataclass
class StakeholderConfig:
    """Configuration for stakeholder-specific features"""
    
    # Federal Policy Makers
    federal_features: List[str] = field(default_factory=lambda: [
        'national_trends',
        'interstate_transmission',
        'resource_allocation',
        'policy_impact_analysis',
        'economic_modeling',
        'variant_detection'
    ])
    
    # Healthcare Professionals  
    healthcare_features: List[str] = field(default_factory=lambda: [
        'local_outbreak_alerts',
        'capacity_planning',
        'clinical_indicators',
        'patient_flow_prediction',
        'supply_chain_monitoring',
        'peer_communication'
    ])
    
    # State Governments
    state_features: List[str] = field(default_factory=lambda: [
        'state_level_trends',
        'county_breakdown',
        'resource_management',
        'public_communication_support',
        'interstate_coordination',
        'economic_impact_analysis'
    ])
    
    # Alert thresholds by stakeholder
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'federal': {
            'hospitalization_increase': 0.25,  # 25% increase
            'interstate_transmission_risk': 0.7,
            'resource_shortage_risk': 0.8
        },
        'healthcare': {
            'local_outbreak_probability': 0.6,
            'capacity_utilization': 0.85,
            'supply_shortage_risk': 0.7
        },
        'state': {
            'state_outbreak_probability': 0.5,
            'county_alert_threshold': 0.6,
            'public_concern_level': 0.8
        }
    })

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = True
    log_level: str = "INFO"
    
    # Data directories
    data_dir: str = "/home/ubuntu/disease_early_warning_system/data"
    model_dir: str = "/home/ubuntu/disease_early_warning_system/models"
    log_dir: str = "/home/ubuntu/disease_early_warning_system/logs"
    
    # Database settings
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None
    
    # Security
    secret_key: Optional[str] = None
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    # Component configurations
    cdc: CDCConfig = field(default_factory=CDCConfig)
    social_media: SocialMediaConfig = field(default_factory=SocialMediaConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    stakeholders: StakeholderConfig = field(default_factory=StakeholderConfig)

class ConfigManager:
    """Manages system configuration with environment variable overrides"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = SystemConfig()
        
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
            
        self.load_from_environment()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Update configuration with file data
            self._update_config_from_dict(self.config, config_data)
            
        except Exception as e:
            print(f"Error loading config file {config_file}: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        # CDC configuration
        if os.getenv('CDC_API_KEY'):
            self.config.cdc.api_key = os.getenv('CDC_API_KEY')
            
        # Social media configuration
        if os.getenv('TWITTER_API_KEY'):
            self.config.social_media.twitter_api_key = os.getenv('TWITTER_API_KEY')
        if os.getenv('TWITTER_API_SECRET'):
            self.config.social_media.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        if os.getenv('TWITTER_BEARER_TOKEN'):
            self.config.social_media.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if os.getenv('REDDIT_CLIENT_ID'):
            self.config.social_media.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        if os.getenv('REDDIT_CLIENT_SECRET'):
            self.config.social_media.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            
        # Database configuration
        if os.getenv('DATABASE_URL'):
            self.config.database_url = os.getenv('DATABASE_URL')
        if os.getenv('REDIS_URL'):
            self.config.redis_url = os.getenv('REDIS_URL')
        if os.getenv('ELASTICSEARCH_URL'):
            self.config.elasticsearch_url = os.getenv('ELASTICSEARCH_URL')
            
        # System configuration
        if os.getenv('ENVIRONMENT'):
            self.config.environment = os.getenv('ENVIRONMENT')
        if os.getenv('DEBUG'):
            self.config.debug = os.getenv('DEBUG').lower() == 'true'
        if os.getenv('SECRET_KEY'):
            self.config.secret_key = os.getenv('SECRET_KEY')
    
    def _update_config_from_dict(self, config_obj, config_dict):
        """Recursively update configuration object from dictionary"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                elif hasattr(current_value, '__dict__') and isinstance(value, dict):
                    self._update_config_from_dict(current_value, value)
                else:
                    setattr(config_obj, key, value)
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        try:
            config_dict = self._config_to_dict(self.config)
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def _config_to_dict(self, config_obj):
        """Convert configuration object to dictionary"""
        if hasattr(config_obj, '__dict__'):
            result = {}
            for key, value in config_obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        else:
            return config_obj
    
    def get_stakeholder_config(self, stakeholder_type: str) -> Dict:
        """Get configuration specific to a stakeholder type"""
        stakeholder_configs = {
            'federal': {
                'features': self.config.stakeholders.federal_features,
                'thresholds': self.config.stakeholders.alert_thresholds.get('federal', {}),
                'dashboard_title': 'Federal Policy Dashboard',
                'default_view': 'national'
            },
            'healthcare': {
                'features': self.config.stakeholders.healthcare_features,
                'thresholds': self.config.stakeholders.alert_thresholds.get('healthcare', {}),
                'dashboard_title': 'Healthcare Professional Dashboard',
                'default_view': 'local'
            },
            'state': {
                'features': self.config.stakeholders.state_features,
                'thresholds': self.config.stakeholders.alert_thresholds.get('state', {}),
                'dashboard_title': 'State Government Dashboard',
                'default_view': 'state'
            }
        }
        
        return stakeholder_configs.get(stakeholder_type, {})

# Global configuration instance
config_manager = ConfigManager()

# Convenience function to get configuration
def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config_manager.config

def get_stakeholder_config(stakeholder_type: str) -> Dict:
    """Get stakeholder-specific configuration"""
    return config_manager.get_stakeholder_config(stakeholder_type)

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = get_config()
    
    print("System Configuration:")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Data Directory: {config.data_dir}")
    
    print("\nCDC Configuration:")
    print(f"Base URL: {config.cdc.base_url}")
    print(f"Endpoints: {list(config.cdc.endpoints.keys())}")
    
    print("\nStakeholder Features:")
    for stakeholder in ['federal', 'healthcare', 'state']:
        stakeholder_config = get_stakeholder_config(stakeholder)
        print(f"{stakeholder.title()}: {stakeholder_config.get('features', [])}")
    
    # Save configuration to file
    config_manager.save_to_file("/home/ubuntu/disease_early_warning_system/config/system_config.json")

