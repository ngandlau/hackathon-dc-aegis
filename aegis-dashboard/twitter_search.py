#!/usr/bin/env python3
import os
import sys
import requests
import pandas as pd
from datetime import datetime
import json

health_query = """(
  ("tested positive" (covid OR "covid-19" OR coronavirus))
  OR ("shortness of breath"
       OR anosmia
       OR ageusia
       OR "sore throat"
       OR "stuffy nose"
       OR "loss of smell"
     )
)"""
        
        # Add location filter and other parameters
QUERY = f"{health_query} profile_country:US lang:en -is:retweet"

# Fixed time window: October 1, 2024 → May 31, 2025 (ISO 8601 format)
START_TIME = "2024-10-01T00:00:00Z"
END_TIME   = "2025-05-31T23:59:59Z"

def get_twitter_data(query: str, max_results: int = None) -> tuple:
    """
    Calls Twitter API v2 full-archive search endpoint to retrieve tweets and their counts.
    Returns both the tweet data and the total count.
    If max_results is None, retrieves all available tweets up to 15,000.
    """
    bearer_token = "<TWITTER_BEARER_TOKEN>"
    if not bearer_token:
        print(
            "Error: please set your Twitter Bearer Token in the environment variable TWITTER_BEARER_TOKEN",
            file=sys.stderr
        )
        sys.exit(1)

    # First get the count
    count_url = "https://api.twitter.com/2/tweets/counts/all"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    count_params = {
        "query": query,
        "start_time": START_TIME,
        "end_time": END_TIME,
        "granularity": "day"
    }

    count_response = requests.get(count_url, headers=headers, params=count_params)
    if count_response.status_code != 200:
        print(f"Error: Twitter API returned {count_response.status_code} — {count_response.text}", file=sys.stderr)
        sys.exit(1)

    count_payload = count_response.json()
    data_buckets = count_payload.get("data", [])
    total_count = sum(bucket.get("tweet_count", 0) for bucket in data_buckets)
    print(f"Total available tweets: {total_count}")

    # Set maximum results to 15,000 if not specified
    if max_results is None:
        max_results = 15000
    else:
        max_results = min(max_results, 15000)  # Cap at 15,000 even if higher value is specified

    # Now get the actual tweets
    search_url = "https://api.twitter.com/2/tweets/search/all"
    search_params = {
        "query": query,
        "start_time": START_TIME,
        "end_time": END_TIME,
        "max_results": min(500, max_results),  # Don't request more than total available or max_results
        "tweet.fields": "created_at,public_metrics,geo,entities",
        "user.fields": "location,description",
        "expansions": "author_id,geo.place_id"
    }

    all_tweets = []
    next_token = None
    page_count = 0
    start_time = datetime.now()

    while True:
        if next_token:
            search_params['next_token'] = next_token

        search_response = requests.get(search_url, headers=headers, params=search_params)
        if search_response.status_code != 200:
            print(f"Error: Twitter API returned {search_response.status_code} — {search_response.text}", file=sys.stderr)
            break

        search_payload = search_response.json()
        tweets = search_payload.get('data', [])
        
        # Only add tweets if we haven't exceeded the max_results
        remaining_tweets = max_results - len(all_tweets)
        if remaining_tweets <= 0:
            print("\nReached maximum tweet limit (15,000)")
            break
            
        tweets_to_add = tweets[:remaining_tweets]
        all_tweets.extend(tweets_to_add)
        page_count += 1

        # Calculate progress
        elapsed_time = (datetime.now() - start_time).total_seconds()
        tweets_per_second = len(all_tweets) / elapsed_time if elapsed_time > 0 else 0
        remaining_tweets = max_results - len(all_tweets)
        estimated_time = remaining_tweets / tweets_per_second if tweets_per_second > 0 else 0

        print(f"\rRetrieved {len(all_tweets)} tweets ({len(all_tweets)/max_results*100:.1f}%) "
              f"| {tweets_per_second:.1f} tweets/sec | "
              f"Est. remaining time: {estimated_time/60:.1f} minutes", end="")

        # Get next page token
        next_token = search_payload.get('meta', {}).get('next_token')
        if not next_token:
            print("\nReached end of available tweets")
            break

    print(f"\nCompleted in {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
    return total_count, all_tweets

def save_tweets_to_csv(tweets: list, filename: str):
    """Save tweets to a CSV file with relevant fields"""
    if not tweets:
        print("No tweets to save")
        return

    # Extract relevant fields
    tweet_data = []
    for tweet in tweets:
        tweet_info = {
            'id': tweet['id'],
            'created_at': tweet['created_at'],
            'text': tweet['text'],
            'retweet_count': tweet['public_metrics']['retweet_count'],
            'reply_count': tweet['public_metrics']['reply_count'],
            'like_count': tweet['public_metrics']['like_count'],
            'quote_count': tweet['public_metrics']['quote_count']
        }
        
        # Add location if available
        if 'geo' in tweet and 'place_id' in tweet['geo']:
            tweet_info['location'] = tweet['geo']['place_id']
        
        # Add hashtags if available
        if 'entities' in tweet and 'hashtags' in tweet['entities']:
            tweet_info['hashtags'] = ' '.join([tag['tag'] for tag in tweet['entities']['hashtags']])
        
        tweet_data.append(tweet_info)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(tweet_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(tweet_data)} tweets to {filename}")

if __name__ == "__main__":
    print(f"Fetching tweets for query: {QUERY}")
    print(f"Time range: {START_TIME} to {END_TIME}")
    
    # Set max_results to 15000
    total_count, tweets = get_twitter_data(QUERY, max_results=15000)
    print(f"\nTotal matching Tweets: {total_count}")
    print(f"Retrieved {len(tweets)} tweets")
    
    # Save tweets to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"twitter_data_{timestamp}.csv"
    save_tweets_to_csv(tweets, filename)
    
    # Save raw JSON for backup
    with open(f"twitter_data_{timestamp}.json", 'w') as f:
        json.dump(tweets, f, indent=2)
    print(f"Saved raw JSON data to twitter_data_{timestamp}.json")
