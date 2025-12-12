import unittest

from src.news_api import fetch_news_for_ticker


class TestNewsAPI(unittest.TestCase):
    def test_fetch_news_no_key_returns_empty_list(self):
        articles = fetch_news_for_ticker("TSLA", None)
        self.assertIsInstance(articles, list)
        self.assertEqual(articles, [])

