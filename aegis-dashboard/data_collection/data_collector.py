"""
Early Warning Disease Prediction System - Data Collection Module
================================================================

This module handles data collection from multiple sources including:
- CDC hospitalization data via official APIs
- Social media data simulation (Twitter/X and Reddit)
- Supplementary data sources (Google Trends, mobility data)


Date: June 3, 2025
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for data sources"""
    name: str
    url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    enabled: bool = True

class CDCDataCollector:
    """
    Collects hospitalization and COVID-19 data from CDC APIs
    """
    
    def __init__(self, config_path: str = None):
        self.base_url = "https://data.cdc.gov/resource"
        self.endpoints = {
            'hospitalization': '39z2-9zu6.json',  # COVID-19 Hospitalization Metrics
            'case_surveillance': 'vbim-akqf.json',  # COVID-19 Case Surveillance
            'weekly_metrics': 'akn2-qxic.json'  # Weekly Hospitalization Metrics
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Disease-Early-Warning-System/1.0',
            'Accept': 'application/json'
        })
        
    def get_hospitalization_data(self, 
                                start_date: str = None, 
                                end_date: str = None,
                                state: str = None,
                                limit: int = 10000) -> pd.DataFrame:
        """
        Fetch hospitalization data from CDC API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            state: State abbreviation (e.g., 'CA', 'NY')
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with hospitalization data
        """
        try:
            url = f"{self.base_url}/{self.endpoints['hospitalization']}"
            params = {'$limit': limit}
            
            # Add date filters if provided
            if start_date and end_date:
                params['$where'] = f"date between '{start_date}T00:00:00.000' and '{end_date}T23:59:59.999'"
            elif start_date:
                params['$where'] = f"date >= '{start_date}T00:00:00.000'"
                
            # Add state filter if provided
            if state:
                state_filter = f"jurisdiction = '{state}'"
                if '$where' in params:
                    params['$where'] += f" AND {state_filter}"
                else:
                    params['$where'] = state_filter
                    
            logger.info(f"Fetching CDC hospitalization data with params: {params}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Clean and standardize data
                df = self._clean_hospitalization_data(df)
                logger.info(f"Successfully fetched {len(df)} hospitalization records")
            else:
                logger.warning("No hospitalization data returned from CDC API")
                
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CDC hospitalization data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in get_hospitalization_data: {e}")
            return pd.DataFrame()
    
    def _clean_hospitalization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize CDC hospitalization data"""
        try:
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
            # Convert numeric columns
            numeric_columns = [
                'new_covid_19_hospital_admissions',
                'new_covid_19_hospital_admissions_7_day_average',
                'total_hospitalized_covid_19_patients',
                'covid_19_inpatient_bed_occupancy_7_day_average',
                'covid_19_icu_bed_occupancy_7_day_average'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Sort by date and jurisdiction
            if 'date' in df.columns and 'jurisdiction' in df.columns:
                df = df.sort_values(['jurisdiction', 'date'])
                
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning hospitalization data: {e}")
            return df

class SocialMediaSimulator:
    """
    Simulates social media data for Twitter/X and Reddit
    Based on realistic patterns observed during COVID-19 outbreaks
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Symptom keywords and their relative frequencies
        self.covid_symptoms = {
            'fever': 0.3, 'cough': 0.4, 'fatigue': 0.25, 'headache': 0.2,
            'sore throat': 0.15, 'loss of taste': 0.1, 'loss of smell': 0.1,
            'shortness of breath': 0.08, 'body aches': 0.18, 'congestion': 0.12
        }
        
        # Healthcare seeking behavior keywords
        self.healthcare_keywords = {
            'hospital': 0.05, 'doctor': 0.15, 'urgent care': 0.08, 'emergency room': 0.03,
            'test positive': 0.12, 'covid test': 0.2, 'quarantine': 0.1, 'isolation': 0.08
        }
        
        # Sentiment patterns (negative, neutral, positive)
        self.sentiment_weights = [0.4, 0.35, 0.25]
        
        # Geographic distribution (simplified US states)
        self.states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        self.state_populations = {
            'CA': 39538223, 'TX': 29145505, 'FL': 21538187, 'NY': 20201249,
            'PA': 13002700, 'IL': 12812508, 'OH': 11799448, 'GA': 10711908,
            'NC': 10439388, 'MI': 10037261
        }
        
    def generate_twitter_data(self, 
                            start_date: str,
                            end_date: str,
                            outbreak_intensity: float = 1.0) -> pd.DataFrame:
        """
        Generate simulated Twitter/X data for the specified date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            outbreak_intensity: Multiplier for outbreak activity (1.0 = normal, >1.0 = outbreak)
            
        Returns:
            DataFrame with simulated Twitter data
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            tweets = []
            current_date = start_dt
            
            while current_date <= end_dt:
                # Base tweet volume (varies by day of week)
                base_volume = 1000 if current_date.weekday() < 5 else 600  # Weekday vs weekend
                daily_volume = int(base_volume * outbreak_intensity * random.uniform(0.8, 1.2))
                
                for _ in range(daily_volume):
                    tweet = self._generate_single_tweet(current_date, outbreak_intensity)
                    tweets.append(tweet)
                    
                current_date += timedelta(days=1)
                
            df = pd.DataFrame(tweets)
            logger.info(f"Generated {len(df)} simulated tweets from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Twitter data: {e}")
            return pd.DataFrame()
    
    def _generate_single_tweet(self, date: datetime, outbreak_intensity: float) -> Dict:
        """Generate a single simulated tweet"""
        # Determine tweet type based on outbreak intensity
        if random.random() < 0.3 * outbreak_intensity:
            tweet_type = 'symptom'
            content, keywords = self._generate_symptom_tweet()
        elif random.random() < 0.2 * outbreak_intensity:
            tweet_type = 'healthcare'
            content, keywords = self._generate_healthcare_tweet()
        else:
            tweet_type = 'general'
            content, keywords = self._generate_general_tweet()
            
        # Assign sentiment
        sentiment = np.random.choice(['negative', 'neutral', 'positive'], p=self.sentiment_weights)
        
        # Assign location
        state = np.random.choice(list(self.state_populations.keys()), 
                               p=[pop/sum(self.state_populations.values()) 
                                 for pop in self.state_populations.values()])
        
        return {
            'date': date,
            'content': content,
            'tweet_type': tweet_type,
            'keywords': keywords,
            'sentiment': sentiment,
            'state': state,
            'engagement_score': random.uniform(0, 100),
            'retweet_count': max(0, int(np.random.exponential(5) * outbreak_intensity)),
            'like_count': max(0, int(np.random.exponential(10) * outbreak_intensity))
        }
    
    def _generate_symptom_tweet(self) -> Tuple[str, List[str]]:
        """Generate a tweet about COVID symptoms"""
        symptom = np.random.choice(list(self.covid_symptoms.keys()), 
                                 p=list(self.covid_symptoms.values()))
        
        templates = [
            f"Been having {symptom} for the past few days, hope it's not covid",
            f"Woke up with {symptom} today, getting tested just in case",
            f"Anyone else experiencing {symptom}? Seems like it's going around",
            f"Day 3 of {symptom}, finally going to see a doctor",
            f"This {symptom} is really getting to me, staying home today"
        ]
        
        content = random.choice(templates)
        keywords = [symptom, 'covid', 'symptoms']
        
        return content, keywords
    
    def _generate_healthcare_tweet(self) -> Tuple[str, List[str]]:
        """Generate a tweet about healthcare seeking behavior"""
        action = np.random.choice(list(self.healthcare_keywords.keys()),
                                p=list(self.healthcare_keywords.values()))
        
        templates = [
            f"Just got back from the {action}, waiting for results",
            f"Heading to {action} today, wish me luck",
            f"Long wait at {action} today, seems busier than usual",
            f"Finally got my {action} appointment scheduled",
            f"PSA: {action} is really backed up right now, plan ahead"
        ]
        
        content = random.choice(templates)
        keywords = [action, 'healthcare', 'medical']
        
        return content, keywords
    
    def _generate_general_tweet(self) -> Tuple[str, List[str]]:
        """Generate a general health-related tweet"""
        templates = [
            "Staying healthy and taking precautions these days",
            "Remember to wash your hands and stay safe everyone",
            "Anyone else feeling like everyone's getting sick lately?",
            "Taking extra vitamins and staying hydrated",
            "Work from home day, not feeling 100%"
        ]
        
        content = random.choice(templates)
        keywords = ['health', 'wellness', 'prevention']
        
        return content, keywords
    
    def generate_reddit_data(self,
                           start_date: str,
                           end_date: str,
                           outbreak_intensity: float = 1.0) -> pd.DataFrame:
        """
        Generate simulated Reddit data for health-related subreddits
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            outbreak_intensity: Multiplier for outbreak activity
            
        Returns:
            DataFrame with simulated Reddit data
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            subreddits = ['COVID19', 'medicine', 'AskDocs', 'HealthAnxiety', 'coronavirus']
            posts = []
            current_date = start_dt
            
            while current_date <= end_dt:
                # Generate posts for each subreddit
                for subreddit in subreddits:
                    daily_posts = int(50 * outbreak_intensity * random.uniform(0.5, 1.5))
                    
                    for _ in range(daily_posts):
                        post = self._generate_reddit_post(current_date, subreddit, outbreak_intensity)
                        posts.append(post)
                        
                current_date += timedelta(days=1)
                
            df = pd.DataFrame(posts)
            logger.info(f"Generated {len(df)} simulated Reddit posts from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Reddit data: {e}")
            return pd.DataFrame()
    
    def _generate_reddit_post(self, date: datetime, subreddit: str, outbreak_intensity: float) -> Dict:
        """Generate a single simulated Reddit post"""
        # Post types vary by subreddit
        if subreddit in ['COVID19', 'coronavirus']:
            post_types = ['symptom_question', 'test_result', 'policy_discussion', 'personal_experience']
        elif subreddit == 'AskDocs':
            post_types = ['symptom_question', 'medical_advice']
        else:
            post_types = ['general_health', 'anxiety', 'symptom_question']
            
        post_type = random.choice(post_types)
        title, content, keywords = self._generate_reddit_content(post_type, subreddit)
        
        return {
            'date': date,
            'subreddit': subreddit,
            'title': title,
            'content': content,
            'post_type': post_type,
            'keywords': keywords,
            'upvotes': max(0, int(np.random.exponential(20) * outbreak_intensity)),
            'comments': max(0, int(np.random.exponential(5) * outbreak_intensity)),
            'engagement_score': random.uniform(0, 100)
        }
    
    def _generate_reddit_content(self, post_type: str, subreddit: str) -> Tuple[str, str, List[str]]:
        """Generate Reddit post title and content based on type"""
        if post_type == 'symptom_question':
            symptom = random.choice(list(self.covid_symptoms.keys()))
            title = f"[Question] {symptom.title()} for 3 days - should I be concerned?"
            content = f"I've been experiencing {symptom} for the past few days along with some general fatigue. Has anyone else had similar symptoms recently? Should I get tested?"
            keywords = [symptom, 'symptoms', 'question', 'advice']
            
        elif post_type == 'test_result':
            result = random.choice(['positive', 'negative'])
            title = f"Test came back {result} - what now?"
            content = f"Just got my test results back and I'm {result}. Looking for advice on next steps and what to expect."
            keywords = ['test', result, 'results', 'advice']
            
        elif post_type == 'personal_experience':
            title = "My COVID experience - timeline and symptoms"
            content = "Thought I'd share my experience for others who might be going through something similar. Day 1: mild headache..."
            keywords = ['experience', 'timeline', 'symptoms', 'recovery']
            
        else:  # general_health
            title = "General health anxiety during these times"
            content = "Anyone else feeling more anxious about health lately? Every little symptom makes me worry."
            keywords = ['anxiety', 'health', 'worry', 'mental health']
            
        return title, content, keywords

class DataPipeline:
    """
    Orchestrates data collection from all sources and manages data storage
    """
    
    def __init__(self, data_dir: str = "/home/ubuntu/disease_early_warning_system/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.simulated_dir = self.data_dir / "simulated"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.simulated_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.cdc_collector = CDCDataCollector()
        self.social_simulator = SocialMediaSimulator()
        
    def collect_all_data(self, 
                        start_date: str,
                        end_date: str,
                        outbreak_intensity: float = 1.0,
                        use_simulation: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources for the specified date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            outbreak_intensity: Multiplier for outbreak simulation
            use_simulation: Whether to use simulated social media data
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Starting data collection for {start_date} to {end_date}")
        
        data = {}
        
        # Collect CDC hospitalization data
        logger.info("Collecting CDC hospitalization data...")
        cdc_data = self.cdc_collector.get_hospitalization_data(start_date, end_date)
        data['cdc_hospitalization'] = cdc_data
        
        # Save CDC data
        if not cdc_data.empty:
            cdc_file = self.raw_dir / f"cdc_hospitalization_{start_date}_{end_date}.csv"
            cdc_data.to_csv(cdc_file, index=False)
            logger.info(f"Saved CDC data to {cdc_file}")
        
        if use_simulation:
            # Generate simulated social media data
            logger.info("Generating simulated Twitter data...")
            twitter_data = self.social_simulator.generate_twitter_data(start_date, end_date, outbreak_intensity)
            data['twitter'] = twitter_data
            
            logger.info("Generating simulated Reddit data...")
            reddit_data = self.social_simulator.generate_reddit_data(start_date, end_date, outbreak_intensity)
            data['reddit'] = reddit_data
            
            # Save simulated data
            if not twitter_data.empty:
                twitter_file = self.simulated_dir / f"twitter_{start_date}_{end_date}.csv"
                twitter_data.to_csv(twitter_file, index=False)
                logger.info(f"Saved Twitter simulation to {twitter_file}")
                
            if not reddit_data.empty:
                reddit_file = self.simulated_dir / f"reddit_{start_date}_{end_date}.csv"
                reddit_data.to_csv(reddit_file, index=False)
                logger.info(f"Saved Reddit simulation to {reddit_file}")
        
        logger.info("Data collection completed successfully")
        return data
    
    def load_existing_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load previously collected data for the specified date range"""
        data = {}
        
        # Load CDC data
        cdc_file = self.raw_dir / f"cdc_hospitalization_{start_date}_{end_date}.csv"
        if cdc_file.exists():
            data['cdc_hospitalization'] = pd.read_csv(cdc_file)
            logger.info(f"Loaded CDC data from {cdc_file}")
        
        # Load simulated social media data
        twitter_file = self.simulated_dir / f"twitter_{start_date}_{end_date}.csv"
        if twitter_file.exists():
            data['twitter'] = pd.read_csv(twitter_file)
            logger.info(f"Loaded Twitter data from {twitter_file}")
            
        reddit_file = self.simulated_dir / f"reddit_{start_date}_{end_date}.csv"
        if reddit_file.exists():
            data['reddit'] = pd.read_csv(reddit_file)
            logger.info(f"Loaded Reddit data from {reddit_file}")
            
        return data

# Example usage and testing
if __name__ == "__main__":
    # Initialize data pipeline
    pipeline = DataPipeline()
    
    # Collect data for Omicron outbreak period (Dec 2021 - Mar 2022)
    start_date = "2021-12-01"
    end_date = "2022-03-31"
    
    # Collect all data with high outbreak intensity for Omicron period
    data = pipeline.collect_all_data(
        start_date=start_date,
        end_date=end_date,
        outbreak_intensity=2.5,  # High intensity for Omicron outbreak
        use_simulation=True
    )
    
    # Print summary statistics
    for source, df in data.items():
        if not df.empty:
            print(f"\n{source.upper()} Data Summary:")
            print(f"Records: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Columns: {list(df.columns)}")
        else:
            print(f"\n{source.upper()}: No data collected")

