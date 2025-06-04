import json
import logging
import os
from typing import Dict, List

from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Utility: Load tweets from JSON file
def load_tweets(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        data = json.load(f)
    # Assume data is a list of dicts with 'text' or 'full_text' field
    tweets = [tweet.get("text") or tweet.get("full_text", "") for tweet in data]
    return [t for t in tweets if t][:500]


# Utility: Chunk tweets for LLM context window
def chunk_tweets(tweets: List[str], max_tokens: int = 30000) -> List[List[str]]:
    # Simple chunking by tweet count (not exact tokens)
    chunk_size = 100  # Adjust as needed for context window
    return [tweets[i : i + chunk_size] for i in range(0, len(tweets), chunk_size)]


# Performer: Calls Anthropic Claude API
class LLMPerformer:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("CLAUDE_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def analyze(self, tweets: List[str], prompt: str) -> List[Dict]:
        results = []
        tweet_chunks = chunk_tweets(tweets)
        for idx, chunk in enumerate(tweet_chunks):
            chunk_text = "\n".join(chunk)
            full_prompt = (
                f"{prompt}\n\nTWEETS:\n{chunk_text}\n\nReturn the result as JSON."
            )
            logger.info(
                f"Calling Claude API for chunk {idx + 1}/{len(tweet_chunks)}..."
            )
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.3,
                system="You are a social media analyst. Be concise and data-driven.",
                messages=[{"role": "user", "content": full_prompt}],
            )
            # Try to parse JSON from response
            try:
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.lstrip("`").strip()
                    if text.startswith("json") or text.startswith("html"):
                        text = text.split("\n", 1)[1] if "\n" in text else text
                result = json.loads(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                results.append(
                    {"error": str(e), "raw_response": response.content[0].text}
                )
        return results


# Controllers
class TrendController:
    name = "trends"

    def __init__(self, tweets: List[str], llm: LLMPerformer):
        self.tweets = tweets
        self.llm = llm

    def run(self):
        prompt = "Given these tweets, what are the top 5 trending topics, hashtags, or keywords? Return as a JSON list under the key 'trends'."
        return self.llm.analyze(self.tweets, prompt)


class SentimentController:
    name = "sentiment"

    def __init__(self, tweets: List[str], llm: LLMPerformer):
        self.tweets = tweets
        self.llm = llm

    def run(self):
        prompt = "Analyze the sentiment of these tweets. What is the overall mood? Any spikes in negativity or positivity? Return as JSON with keys 'overall_sentiment', 'summary', and 'notable_examples'."
        return self.llm.analyze(self.tweets, prompt)


class SymptomController:
    name = "symptoms"

    def __init__(self, tweets: List[str], llm: LLMPerformer):
        self.tweets = tweets
        self.llm = llm

    def run(self):
        prompt = "Extract mentions of symptoms or health concerns from these tweets. List the most common ones as a JSON list under the key 'symptoms'."
        return self.llm.analyze(self.tweets, prompt)


# Manager
class SocialDataManager:
    def __init__(self, tweets: List[str], api_key=None):
        if api_key is None:
            api_key = os.getenv("CLAUDE_API_KEY")
        self.llm = LLMPerformer(api_key)
        self.controllers = [
            TrendController(tweets, self.llm),
            SentimentController(tweets, self.llm),
            SymptomController(tweets, self.llm),
        ]

    def run(self):
        results = {}
        for controller in self.controllers:
            logger.info(f"Running {controller.name} analysis...")
            results[controller.name] = controller.run()
        return results


# Example usage (to be run as a script or imported)
if __name__ == "__main__":
    # Set your Anthropic Claude API key here
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    TWEET_JSON_PATH = "data/collected/twitter_data_20250603_233904.json"
    tweets = load_tweets(TWEET_JSON_PATH)
    agent = SocialDataManager(tweets, CLAUDE_API_KEY)
    analysis_results = agent.run()
    # Save or print results
    with open("data/collected/social_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(json.dumps(analysis_results, indent=2))
