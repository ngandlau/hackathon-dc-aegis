#!/usr/bin/env python3
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

def load_twitter_data(file_path):
    """Load Twitter data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_weekly_volumes(tweets):
    """Analyze tweet volumes by week"""
    # Convert to DataFrame
    df = pd.DataFrame(tweets)
    
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Extract week number and year
    df['year'] = df['created_at'].dt.isocalendar().year
    df['week'] = df['created_at'].dt.isocalendar().week
    
    # Group by year and week
    weekly_counts = df.groupby(['year', 'week']).size().reset_index(name='tweet_count')
    
    # Create a proper date for each week (start of week)
    weekly_counts['date'] = weekly_counts.apply(
        lambda x: datetime.fromisocalendar(x['year'], x['week'], 1), 
        axis=1
    )
    
    return weekly_counts

def create_volume_chart(weekly_data):
    """Create an interactive volume chart using Plotly"""
    # Create the figure
    fig = go.Figure()
    
    # Add the volume bars
    fig.add_trace(go.Bar(
        x=weekly_data['date'],
        y=weekly_data['tweet_count'],
        name='Tweet Volume',
        marker_color='#1DA1F2',  # Twitter blue
        opacity=0.8
    ))
    
    # Add a trend line
    fig.add_trace(go.Scatter(
        x=weekly_data['date'],
        y=weekly_data['tweet_count'].rolling(window=2).mean(),
        name='Trend',
        line=dict(color='#E0245E', width=2),  # Twitter red
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title='Weekly Twitter Volume Analysis',
        xaxis_title='Week',
        yaxis_title='Number of Tweets',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text="Source: Twitter API",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.1
            )
        ]
    )
    
    # Update axes
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        tickangle=45,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    return fig

def main():
    # Load the data
    file_path = '/Users/shashankchikara/Downloads/disease_early_warning_system/twitter_data_20250603_233904.json'
    tweets = load_twitter_data(file_path)
    
    # Analyze weekly volumes
    weekly_data = analyze_weekly_volumes(tweets)
    
    # Print summary statistics
    print("\nWeekly Volume Summary:")
    print(f"Total weeks analyzed: {len(weekly_data)}")
    print(f"Total tweets: {weekly_data['tweet_count'].sum():,}")
    print(f"Average tweets per week: {weekly_data['tweet_count'].mean():.1f}")
    print(f"Maximum weekly volume: {weekly_data['tweet_count'].max():,}")
    print(f"Minimum weekly volume: {weekly_data['tweet_count'].min():,}")
    
    # Create and save the chart
    fig = create_volume_chart(weekly_data)
    fig.write_html('twitter_volume_analysis.html')
    print("\nChart has been saved as 'twitter_volume_analysis.html'")
    
    # Save the weekly data to CSV
    weekly_data.to_csv('twitter_weekly_volumes.csv', index=False)
    print("Weekly volume data has been saved to 'twitter_weekly_volumes.csv'")

if __name__ == "__main__":
    main() 