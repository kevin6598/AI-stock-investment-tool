"""
Strategy Candidates Routes
---------------------------
GET /api/v1/strategy/candidates - Strategy-filtered stock candidates
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas import StrategyCandidateSchema, StrategyCandidatesResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/strategy", tags=["candidates"])

# Injected at startup by main.py
_model_cache = None
_feature_pipeline = None


def set_dependencies(model_cache, feature_pipeline):
    global _model_cache, _feature_pipeline
    _model_cache = model_cache
    _feature_pipeline = feature_pipeline


@router.get("/candidates", response_model=StrategyCandidatesResponse)
async def get_strategy_candidates(
    horizon: str = Query(default="3M", pattern="^(1M|3M|6M|1Y)$"),
    allocation: str = Query(default="risk_parity"),
    market: str = Query(default="US"),
):
    """Get stock candidates filtered by the surviving strategy signal.

    Runs StrategyFilter to find tickers matching
    mom_60d decile 4 AND high_52w_pct decile 0,
    then runs Top10Engine on the narrowed universe.
    """
    try:
        from engine.strategy_filter import get_strategy_filter
        from engine.top10 import Top10Engine
        from api.routes.predict import _predict_single
        from strategy_definition.us_63d_mom_60d_d4_52w_d0 import IDENTITY

        # Step 1: Run strategy filter
        sf = get_strategy_filter()
        filter_result = sf.filter(market=market.upper())

        # Step 2: Run Top10 with strategy filter
        model_version = None
        if _model_cache is not None:
            model_version = getattr(_model_cache, "model_version", None)

        engine = Top10Engine(
            predict_fn=_predict_single,
            model_version=model_version,
        )
        result = engine.select(
            market=market.upper(),
            horizon=horizon,
            allocation_mode=allocation,
            strategy_filter=filter_result,
        )

        # Build response with strategy feature fields
        stocks = []
        for s in result.stocks:
            stocks.append(StrategyCandidateSchema(
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
                mom_60d=s.strategy_mom_60d,
                high_52w_pct=s.strategy_high_52w_pct,
                mom_60d_decile=s.strategy_mom_60d_decile,
                high_52w_pct_decile=s.strategy_high_52w_pct_decile,
            ))

        strategy_name = (
            "mom_60d d%d AND high_52w_pct d%d"
            % (IDENTITY.feature_1_decile, IDENTITY.feature_2_decile)
        )

        return StrategyCandidatesResponse(
            strategy_id=IDENTITY.strategy_id,
            strategy_name=strategy_name,
            market=result.market,
            horizon=result.horizon,
            stocks=stocks,
            generated_at=result.generated_at,
            model_version=result.model_version,
            universe_size=filter_result.universe_size,
            signal_matches=len(filter_result.candidates),
            pass_rate=result.pass_rate,
        )

    except Exception as e:
        logger.error("Strategy candidates generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Strategy candidates generation failed: %s" % str(e),
        )
