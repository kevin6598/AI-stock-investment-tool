"""
Sentiment Routes
-----------------
GET /api/v1/sentiment/{ticker} - Sentiment analysis
"""

import logging
from fastapi import APIRouter, HTTPException

from api.schemas import SentimentResponse, SentimentScore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["sentiment"])


@router.get("/sentiment/{ticker}", response_model=SentimentResponse)
async def get_sentiment(ticker: str):
    """Get sentiment analysis for a ticker."""
    try:
        from models.sentiment import get_extended_sentiment_features

        features = get_extended_sentiment_features(ticker.upper())

        sentiment = SentimentScore(
            sentiment_mean=round(features.get("sentiment_mean", 0.0), 4),
            sentiment_weighted=round(features.get("sentiment_weighted", 0.0), 4),
            positive_ratio=round(features.get("positive_ratio", 0.33), 4),
            negative_ratio=round(features.get("negative_ratio", 0.33), 4),
            news_volume=int(features.get("news_volume", 0)),
            sentiment_momentum=round(features.get("sentiment_momentum", 0.0), 4),
            event_direction=round(features.get("event_direction", 0.0), 4),
            event_magnitude=round(features.get("event_magnitude", 0.0), 4),
            macro_impact=round(features.get("macro_impact", 0.0), 4),
        )

        keywords = {
            k: round(v, 4) for k, v in features.items()
            if k.startswith("kw_")
        }

        return SentimentResponse(
            ticker=ticker.upper(),
            sentiment=sentiment,
            keywords=keywords,
        )

    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
