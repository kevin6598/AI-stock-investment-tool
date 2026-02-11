"""
Top 10 Routes
--------------
GET /api/v1/portfolio/top10/{market} - Market-specific top 10 stock picks
"""

from typing import Optional
import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas import Top10Response, Top10StockSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/portfolio", tags=["top10"])

# Injected at startup by main.py
_model_cache = None
_feature_pipeline = None


def set_dependencies(model_cache, feature_pipeline):
    global _model_cache, _feature_pipeline
    _model_cache = model_cache
    _feature_pipeline = feature_pipeline


@router.get("/top10/{market}", response_model=Top10Response)
async def get_top10(
    market: str,
    horizon: str = Query(default="1M", pattern="^(1M|3M|6M|1Y)$"),
    allocation: str = Query(default="risk_parity"),
):
    """Get Top 10 stock picks for a specific market.

    Args:
        market: "US", "KR", or "ALL".
        horizon: Forecast horizon (1M, 3M, 6M, 1Y).
        allocation: Weight allocation mode (equal, risk_parity).
    """
    market_upper = market.upper()
    if market_upper not in ("US", "KR", "ALL"):
        raise HTTPException(
            status_code=400,
            detail="market must be US, KR, or ALL",
        )

    try:
        from engine.top10 import Top10Engine
        from api.routes.predict import _predict_single

        model_version = None
        if _model_cache is not None:
            model_version = getattr(_model_cache, "model_version", None)

        engine = Top10Engine(
            predict_fn=_predict_single,
            model_version=model_version,
        )
        result = engine.select(
            market=market_upper,
            horizon=horizon,
            allocation_mode=allocation,
        )

        return Top10Response(
            market=result.market,
            horizon=result.horizon,
            stocks=[
                Top10StockSchema(
                    rank=s.rank,
                    ticker=s.ticker,
                    score=s.score,
                    direction=s.direction,
                    p_up=s.p_up,
                    expected_return=s.expected_return,
                    confidence=s.confidence,
                    risk_score=s.risk_score,
                    sentiment_score=s.sentiment_score,
                    allocation_weight=s.allocation_weight,
                    reasons=s.reasons,
                )
                for s in result.stocks
            ],
            generated_at=result.generated_at,
            model_version=result.model_version,
            total_candidates=result.total_candidates,
            pass_rate=result.pass_rate,
        )

    except Exception as e:
        logger.error("Top 10 generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Top 10 generation failed: %s" % str(e))
