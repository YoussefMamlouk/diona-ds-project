"""News API helpers.

Provide a small helper to fetch top news articles for a ticker using NewsAPI.
The function is resilient to missing API keys and returns an empty list if
news cannot be retrieved.
"""
from typing import List, Dict, Optional
import requests


def fetch_news_for_ticker(ticker: str, api_key: Optional[str]) -> List[Dict[str, object]]:
	"""Fetch recent news articles for `ticker` using NewsAPI.org.

	Returns a list of article dicts (may be empty). Caller should handle
	network errors gracefully.
	"""
	if not api_key:
		return []
	try:
		url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
		resp = requests.get(url, timeout=10)
		if resp.status_code != 200:
			return []
		data = resp.json()
		return data.get("articles", [])
	except Exception:
		return []

