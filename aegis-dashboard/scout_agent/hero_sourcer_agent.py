import os
import time
import json
from datetime import datetime
import logging
import requests
from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performer: Uses Firecrawl MCP API to search and synthesize
class FirecrawlSearchPerformer:
    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, query, context="health"):
        url = "https://api.firecrawl.dev/mcp/firecrawl/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": f"latest {context} information related to: {query}",
            "limit": 5,
            "lang": "en",
            "country": "us",
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True
            }
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data
        except Exception as e:
            logger.warning(f"Firecrawl MCP search failed: {e}")
            return {"error": str(e)}

# Performer: Uses Claude API to summarize Firecrawl results
class ClaudeSummarizerPerformer:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
    def summarize(self, firecrawl_data, context="health"):
        prompt = (
            f"Given the following web search results about {context}, "
            "summarize the top 5 key findings in clear, concise bullet points.\n\n"
            f"Results:\n{json.dumps(firecrawl_data, indent=2)}"
        )
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class DataSaverPerformer:
    def save(self, data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filename}")

# Controllers
class NewsController:
    def __init__(self, searcher, summarizer, saver):
        self.searcher = searcher
        self.summarizer = summarizer
        self.saver = saver
    def run(self):
        data = self.searcher.search("latest US public health news")
        summary = self.summarizer.summarize(data, context="US public health news")
        fname = f"data/collected/news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        self.saver.save({"raw": data, "summary": summary}, fname)
        return {"raw": data, "summary": summary}

class ResearchController:
    def __init__(self, searcher, summarizer, saver):
        self.searcher = searcher
        self.summarizer = summarizer
        self.saver = saver
    def run(self):
        data = self.searcher.search("latest medical research preprints")
        summary = self.summarizer.summarize(data, context="medical research preprints")
        fname = f"data/collected/research_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        self.saver.save({"raw": data, "summary": summary}, fname)
        return {"raw": data, "summary": summary}

class SocialController:
    def __init__(self, searcher, summarizer, saver):
        self.searcher = searcher
        self.summarizer = summarizer
        self.saver = saver
    def run(self):
        data = self.searcher.search("trending health topics on social media")
        summary = self.summarizer.summarize(data, context="health topics on social media")
        fname = f"data/collected/social_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        self.saver.save({"raw": data, "summary": summary}, fname)
        return {"raw": data, "summary": summary}

# Manager: Orchestrates everything
class HeroSourcerManager:
    def __init__(self, firecrawl_api_key, claude_api_key):
        self.searcher = FirecrawlSearchPerformer(firecrawl_api_key)
        self.summarizer = ClaudeSummarizerPerformer(claude_api_key)
        self.saver = DataSaverPerformer()
        self.controllers = [
            NewsController(self.searcher, self.summarizer, self.saver),
            ResearchController(self.searcher, self.summarizer, self.saver),
            SocialController(self.searcher, self.summarizer, self.saver)
        ]
    def run_once(self):
        results = {}
        for controller in self.controllers:
            cname = controller.__class__.__name__
            logger.info(f"Running {cname}...")
            results[cname] = controller.run()
        return results
    def run_forever(self, interval_minutes=15):
        while True:
            logger.info(f"Running hero_sourcer at {datetime.now()}")
            self.run_once()
            logger.info(f"Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

# Example usage
if __name__ == "__main__":
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    agent = HeroSourcerManager(FIRECRAWL_API_KEY, CLAUDE_API_KEY)
    agent.run_forever(interval_minutes=15) 