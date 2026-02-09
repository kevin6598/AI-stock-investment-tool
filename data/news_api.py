"""
News API
--------
Fetches financial news for sentiment analysis.

Sources (in priority order):
  1. yfinance Ticker.news (no API key needed)
  2. RSS feeds from major financial outlets (no API key needed)
  3. Placeholder for NewsAPI.org or other premium sources

Returns structured news items with title, source, timestamp, and URL.
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def fetch_yfinance_news(ticker: str, max_items: int = 20) -> List[Dict]:
    """Fetch news from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []

        items = []
        for article in raw_news[:max_items]:
            content = article.get("content", {})
            pub_date = content.get("pubDate")
            title = content.get("title", "")
            if not title:
                title = article.get("title", "")

            items.append({
                "title": title,
                "source": content.get("provider", {}).get("displayName", "yfinance"),
                "timestamp": pub_date if pub_date else datetime.now().isoformat(),
                "url": content.get("canonicalUrl", {}).get("url", ""),
                "ticker": ticker.upper(),
            })
        return items
    except Exception as e:
        logger.warning(f"yfinance news fetch failed for {ticker}: {e}")
        return []


def fetch_rss_news(ticker: str, max_items: int = 10) -> List[Dict]:
    """Fetch news from public RSS feeds."""
    try:
        import feedparser
    except ImportError:
        return []

    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    ]

    items = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_items]:
                pub_date = ""
                if hasattr(entry, "published"):
                    pub_date = entry.published

                items.append({
                    "title": entry.get("title", ""),
                    "source": feed.feed.get("title", "RSS"),
                    "timestamp": pub_date,
                    "url": entry.get("link", ""),
                    "ticker": ticker.upper(),
                })
        except Exception as e:
            logger.warning(f"RSS feed failed: {e}")
            continue

    return items


def fetch_news(ticker: str, max_items: int = 30) -> List[Dict]:
    """Fetch news from all available sources.

    Args:
        ticker: Stock ticker symbol.
        max_items: Maximum total items to return.

    Returns:
        List of dicts with keys: title, source, timestamp, url, ticker.
    """
    all_news = []

    # Try yfinance first (most reliable)
    yf_news = fetch_yfinance_news(ticker, max_items)
    all_news.extend(yf_news)

    # Supplement with RSS if needed
    if len(all_news) < max_items:
        rss_news = fetch_rss_news(ticker, max_items - len(all_news))
        all_news.extend(rss_news)

    # Deduplicate by title
    seen = set()
    unique = []
    for item in all_news:
        title = item["title"].strip().lower()
        if title and title not in seen:
            seen.add(title)
            unique.append(item)

    return unique[:max_items]
