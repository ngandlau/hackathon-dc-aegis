"""
Supplementary Data Collection Module
===================================

Collects additional data sources to enhance disease prediction:
- Google Trends data simulation
- Mobility pattern data simulation  
- Economic indicators simulation
- Weather data simulation

Date: June 3, 2025
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GoogleTrendsSimulator:
    """Simulates Google Trends data for health-related search terms"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Health-related search terms and their baseline popularity
        self.search_terms = {
            'covid symptoms': 20,
            'fever': 15,
            'cough': 18,
            'covid test': 25,
            'hospital near me': 10,
            'urgent care': 12,
            'loss of taste': 8,
            'shortness of breath': 6,
            'covid vaccine': 30,
            'quarantine guidelines': 5
        }
        
    def generate_trends_data(self, 
                           start_date: str,
                           end_date: str,
                           outbreak_intensity: float = 1.0) -> pd.DataFrame:
        """Generate simulated Google Trends data"""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            trends_data = []
            current_date = start_dt
            
            while current_date <= end_dt:
                for term, baseline in self.search_terms.items():
                    # Add seasonal and outbreak effects
                    seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                    outbreak_factor = outbreak_intensity if 'covid' in term or 'symptoms' in term else 1.0
                    
                    # Calculate search volume with noise
                    volume = baseline * seasonal_factor * outbreak_factor * random.uniform(0.7, 1.3)
                    volume = max(0, min(100, volume))  # Clamp to 0-100 range
                    
                    trends_data.append({
                        'date': current_date,
                        'search_term': term,
                        'search_volume': volume,
                        'region': 'US'
                    })
                    
                current_date += timedelta(days=1)
                
            df = pd.DataFrame(trends_data)
            logger.info(f"Generated {len(df)} Google Trends records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Google Trends data: {e}")
            return pd.DataFrame()

class MobilityDataSimulator:
    """Simulates population mobility data similar to Google/Apple mobility reports"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.mobility_categories = [
            'retail_recreation',
            'grocery_pharmacy', 
            'parks',
            'transit_stations',
            'workplaces',
            'residential'
        ]
        
        # Baseline mobility patterns (% change from pre-pandemic baseline)
        self.baseline_patterns = {
            'retail_recreation': -10,
            'grocery_pharmacy': -5,
            'parks': 0,
            'transit_stations': -15,
            'workplaces': -20,
            'residential': 15
        }
        
    def generate_mobility_data(self,
                             start_date: str,
                             end_date: str,
                             outbreak_intensity: float = 1.0) -> pd.DataFrame:
        """Generate simulated mobility data"""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            mobility_data = []
            current_date = start_dt
            
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
            
            while current_date <= end_dt:
                for state in states:
                    for category in self.mobility_categories:
                        baseline = self.baseline_patterns[category]
                        
                        # Add outbreak effects (higher intensity = more mobility reduction)
                        if category == 'residential':
                            outbreak_effect = 10 * (outbreak_intensity - 1.0)  # More time at home
                        else:
                            outbreak_effect = -15 * (outbreak_intensity - 1.0)  # Less mobility
                            
                        # Add day-of-week effects
                        weekday_factor = 1.0 if current_date.weekday() < 5 else 0.7
                        
                        # Calculate mobility change with noise
                        mobility_change = (baseline + outbreak_effect) * weekday_factor
                        mobility_change += random.uniform(-5, 5)  # Add noise
                        
                        mobility_data.append({
                            'date': current_date,
                            'state': state,
                            'category': category,
                            'mobility_change_pct': mobility_change
                        })
                        
                current_date += timedelta(days=1)
                
            df = pd.DataFrame(mobility_data)
            logger.info(f"Generated {len(df)} mobility records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating mobility data: {e}")
            return pd.DataFrame()

class EconomicIndicatorSimulator:
    """Simulates economic indicators that may correlate with disease outbreaks"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_economic_data(self,
                             start_date: str,
                             end_date: str,
                             outbreak_intensity: float = 1.0) -> pd.DataFrame:
        """Generate simulated economic indicator data"""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            economic_data = []
            current_date = start_dt
            
            # Weekly data points (every 7 days)
            while current_date <= end_dt:
                # Unemployment claims (outbreak increases claims)
                base_claims = 250000
                outbreak_effect = 50000 * (outbreak_intensity - 1.0)
                unemployment_claims = base_claims + outbreak_effect + random.uniform(-20000, 20000)
                unemployment_claims = max(0, unemployment_claims)
                
                # Business closures (outbreak increases closures)
                base_closures = 1000
                closure_effect = 500 * (outbreak_intensity - 1.0)
                business_closures = base_closures + closure_effect + random.uniform(-100, 100)
                business_closures = max(0, business_closures)
                
                # Healthcare sector employment
                base_healthcare_jobs = 50000
                healthcare_effect = 5000 * outbreak_intensity  # Outbreak increases healthcare jobs
                healthcare_jobs = base_healthcare_jobs + healthcare_effect + random.uniform(-2000, 2000)
                
                economic_data.append({
                    'date': current_date,
                    'unemployment_claims': unemployment_claims,
                    'business_closures': business_closures,
                    'healthcare_jobs_posted': healthcare_jobs,
                    'consumer_confidence': max(0, min(100, 70 - 10 * (outbreak_intensity - 1.0) + random.uniform(-5, 5)))
                })
                
                current_date += timedelta(days=7)  # Weekly data
                
            df = pd.DataFrame(economic_data)
            logger.info(f"Generated {len(df)} economic indicator records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating economic data: {e}")
            return pd.DataFrame()

class WeatherDataSimulator:
    """Simulates weather data that may influence disease transmission"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Climate data for major US cities
        self.city_climates = {
            'New York': {'temp_range': (32, 85), 'humidity_range': (45, 75)},
            'Los Angeles': {'temp_range': (50, 85), 'humidity_range': (35, 65)},
            'Chicago': {'temp_range': (20, 85), 'humidity_range': (40, 80)},
            'Houston': {'temp_range': (45, 95), 'humidity_range': (55, 85)},
            'Phoenix': {'temp_range': (45, 110), 'humidity_range': (15, 45)},
            'Philadelphia': {'temp_range': (30, 85), 'humidity_range': (45, 75)},
            'San Antonio': {'temp_range': (40, 95), 'humidity_range': (50, 80)},
            'San Diego': {'temp_range': (55, 80), 'humidity_range': (40, 70)},
            'Dallas': {'temp_range': (35, 95), 'humidity_range': (45, 75)},
            'San Jose': {'temp_range': (45, 85), 'humidity_range': (35, 65)}
        }
        
    def generate_weather_data(self,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """Generate simulated weather data"""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            weather_data = []
            current_date = start_dt
            
            while current_date <= end_dt:
                for city, climate in self.city_climates.items():
                    # Seasonal temperature variation
                    day_of_year = current_date.timetuple().tm_yday
                    seasonal_temp_factor = 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 172) / 365))
                    
                    temp_min, temp_max = climate['temp_range']
                    temperature = temp_min + (temp_max - temp_min) * seasonal_temp_factor
                    temperature += random.uniform(-10, 10)  # Daily variation
                    
                    # Humidity with seasonal variation
                    humidity_min, humidity_max = climate['humidity_range']
                    humidity = humidity_min + (humidity_max - humidity_min) * (1 - seasonal_temp_factor * 0.3)
                    humidity += random.uniform(-10, 10)
                    humidity = max(0, min(100, humidity))
                    
                    # UV index (higher in summer)
                    uv_index = max(0, min(11, 6 + 4 * seasonal_temp_factor + random.uniform(-2, 2)))
                    
                    # Air quality index
                    aqi = max(0, min(300, 50 + random.uniform(-20, 50)))
                    
                    weather_data.append({
                        'date': current_date,
                        'city': city,
                        'temperature_f': temperature,
                        'humidity_pct': humidity,
                        'uv_index': uv_index,
                        'air_quality_index': aqi,
                        'precipitation_inches': max(0, random.exponential(0.1))
                    })
                    
                current_date += timedelta(days=1)
                
            df = pd.DataFrame(weather_data)
            logger.info(f"Generated {len(df)} weather records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating weather data: {e}")
            return pd.DataFrame()

class SupplementaryDataPipeline:
    """Orchestrates collection of all supplementary data sources"""
    
    def __init__(self, data_dir: str = "/home/ubuntu/disease_early_warning_system/data"):
        self.data_dir = data_dir
        self.trends_simulator = GoogleTrendsSimulator()
        self.mobility_simulator = MobilityDataSimulator()
        self.economic_simulator = EconomicIndicatorSimulator()
        self.weather_simulator = WeatherDataSimulator()
        
    def collect_all_supplementary_data(self,
                                     start_date: str,
                                     end_date: str,
                                     outbreak_intensity: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Collect all supplementary data sources"""
        logger.info("Collecting supplementary data sources...")
        
        data = {}
        
        # Google Trends data
        logger.info("Generating Google Trends data...")
        data['google_trends'] = self.trends_simulator.generate_trends_data(
            start_date, end_date, outbreak_intensity
        )
        
        # Mobility data
        logger.info("Generating mobility data...")
        data['mobility'] = self.mobility_simulator.generate_mobility_data(
            start_date, end_date, outbreak_intensity
        )
        
        # Economic indicators
        logger.info("Generating economic indicators...")
        data['economic'] = self.economic_simulator.generate_economic_data(
            start_date, end_date, outbreak_intensity
        )
        
        # Weather data
        logger.info("Generating weather data...")
        data['weather'] = self.weather_simulator.generate_weather_data(
            start_date, end_date
        )
        
        # Save all data
        for source, df in data.items():
            if not df.empty:
                file_path = f"{self.data_dir}/simulated/{source}_{start_date}_{end_date}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {source} data to {file_path}")
        
        return data

# Example usage
if __name__ == "__main__":
    pipeline = SupplementaryDataPipeline()
    
    # Generate supplementary data for Omicron period
    data = pipeline.collect_all_supplementary_data(
        start_date="2021-12-01",
        end_date="2022-03-31",
        outbreak_intensity=2.5
    )
    
    # Print summary
    for source, df in data.items():
        print(f"\n{source.upper()} Data Summary:")
        print(f"Records: {len(df)}")
        print(f"Columns: {list(df.columns)}")

