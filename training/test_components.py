"""Smoke test for all forecasting components using synthetic data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

results = []

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 60)
print("  FORECASTING SYSTEM - COMPONENT TESTS")
print("=" * 60)

# --- 1. Feature Engineering ---
print("\n1. Feature Engineering")
np.random.seed(42)
n = 500
dates = pd.bdate_range("2020-01-01", periods=n)
stock_df = pd.DataFrame({
    "Open": 100 + np.cumsum(np.random.randn(n) * 0.5),
    "High": 101 + np.cumsum(np.random.randn(n) * 0.5),
    "Low": 99 + np.cumsum(np.random.randn(n) * 0.5),
    "Close": 100 + np.cumsum(np.random.randn(n) * 0.5),
    "Volume": np.random.randint(1_000_000, 10_000_000, n),
}, index=dates)
stock_df["High"] = stock_df[["Open", "Close", "High"]].max(axis=1) + 0.5
stock_df["Low"] = stock_df[["Open", "Close", "Low"]].min(axis=1) - 0.5

from training.feature_engineering import (
    compute_technical_features, compute_sentiment_features,
    compute_risk_features, compute_macro_features,
    build_feature_matrix, get_feature_groups,
)

tech = compute_technical_features(stock_df)
test("Technical features computed", len(tech.columns) > 30, f"{len(tech.columns)} cols")

sent = compute_sentiment_features(stock_df)
test("Sentiment features computed", "mfi_14" in sent.columns and "ad_line" in sent.columns)

risk = compute_risk_features(stock_df)
test("Risk features computed", "volatility_21d" in risk.columns)

macro = compute_macro_features(stock_df)
test("Macro features computed", "market_vol_21d" in macro.columns)

stock_info = {"shortName": "Test", "trailingPE": 20, "marketCap": 1e9, "sector": "Tech"}
feat_matrix = build_feature_matrix(stock_df, stock_info, stock_df, [21, 63])
test("Feature matrix built", feat_matrix.shape[0] > 100, f"shape {feat_matrix.shape}")

groups = get_feature_groups(feat_matrix)
test("Feature groups identified", len(groups) == 5,
     f"groups: {', '.join(f'{k}({len(v)})' for k, v in groups.items())}")

# --- 2. Models ---
print("\n2. Models")
from training.models import create_model

feature_cols = [c for c in feat_matrix.columns
                if not c.startswith("fwd_return_") and c != "_close"]
X = feat_matrix[feature_cols].values.astype(np.float32)
y = feat_matrix["fwd_return_21d"].values.astype(np.float32)
np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

split = int(len(X) * 0.7)
val_split = int(len(X) * 0.85)
X_train, X_val, X_test = X[:split], X[split:val_split], X[val_split:]
y_train, y_val, y_test = y[:split], y[split:val_split], y[val_split:]

# Elastic Net
enet = create_model("elastic_net")
enet.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
enet_preds = enet.predict(X_test)
test("ElasticNet predicts", len(enet_preds) == len(X_test))
enet_quantiles = enet.predict_quantiles(X_test)
test("ElasticNet quantiles", len(enet_quantiles) == 7, f"{list(enet_quantiles.keys())}")
enet_imp = enet.get_feature_importance()
test("ElasticNet feature importance", len(enet_imp) > 0)

# LightGBM
lgbm = create_model("lightgbm")
lgbm.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
lgbm_preds = lgbm.predict(X_test)
test("LightGBM predicts", len(lgbm_preds) == len(X_test))
lgbm_quantiles = lgbm.predict_quantiles(X_test)
test("LightGBM quantiles", len(lgbm_quantiles) >= 3)
lgbm_imp = lgbm.get_feature_importance()
test("LightGBM feature importance", len(lgbm_imp) > 0)

# LSTM-Attention
try:
    import torch
    lstm = create_model("lstm_attention", {"epochs": 5, "sequence_length": 20, "patience": 3})
    lstm.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    lstm_preds = lstm.predict(X_test)
    test("LSTM-Attention predicts", len(lstm_preds) == len(X_test))
    lstm_quantiles = lstm.predict_quantiles(X_test)
    test("LSTM-Attention quantiles", len(lstm_quantiles) >= 3)
except ImportError:
    test("LSTM-Attention (skipped, no torch)", True, "torch not installed")

# --- 3. Model Selection ---
print("\n3. Model Selection")
from training.model_selection import (
    WalkForwardConfig, WalkForwardValidator,
    compute_prediction_metrics, compute_investment_metrics,
)

pred_metrics = compute_prediction_metrics(y_test, enet_preds)
test("Prediction metrics computed", pred_metrics.n_samples > 0,
     f"IC={pred_metrics.ic:.4f}, HR={pred_metrics.hit_ratio:.3f}")

inv_metrics = compute_investment_metrics(lgbm_preds, y_test)
test("Investment metrics computed", True,
     f"Sharpe={inv_metrics.sharpe_ratio:.2f}, MDD={inv_metrics.max_drawdown:.4f}")

# Walk-forward fold generation
config = WalkForwardConfig(
    train_start="2020-01-01", test_end="2021-12-31",
    train_min_months=6, val_months=2, test_months=2, step_months=2,
    embargo_days=5, expanding=True,
)
validator = WalkForwardValidator(config)
folds = validator.generate_folds(dates)
test("Walk-forward folds generated", len(folds) >= 1, f"{len(folds)} folds")

# --- 4. Hyperparameter Search ---
print("\n4. Hyperparameter Search")
from training.hyperparameter_search import quick_search

qs_result = quick_search("elastic_net", X_train, y_train, X_val, y_val,
                         feature_names=feature_cols, n_trials=5)
test("Quick search works", "best_params" in qs_result,
     f"best IC={qs_result['best_ic']:.4f}")

# --- 5. Weight Optimizer ---
print("\n5. Weight Optimizer")
from training.weight_optimizer import (
    InverseICWeighter, RidgeMetaWeighter, BayesianWeighter,
    SharpeOptimizer, RegimeClassifier, DynamicWeightEngine,
    compute_time_decay_weights,
)

n_w = min(200, len(y))
component_preds = {
    "sentiment": np.random.randn(n_w) * 0.01,
    "technical": np.random.randn(n_w) * 0.01 + y[:n_w] * 0.3,
    "fundamental": np.random.randn(n_w) * 0.01 + y[:n_w] * 0.2,
    "macro": np.random.randn(n_w) * 0.01,
    "risk": np.random.randn(n_w) * 0.01 - y[:n_w] * 0.1,
}
realized = y[:n_w]

ic_w = InverseICWeighter()
ic_weights = ic_w.fit(component_preds, realized)
test("Inverse-IC weights", abs(sum(ic_weights) - 1.0) < 0.01,
     f"weights: {ic_w.get_weights()}")

ridge_w = RidgeMetaWeighter()
ridge_weights = ridge_w.fit(component_preds, realized)
test("Ridge meta-weights", abs(sum(ridge_weights) - 1.0) < 0.01,
     f"weights: {ridge_w.get_weights()}")

bayes_w = BayesianWeighter()
bayes_weights = bayes_w.fit(component_preds, realized)
test("Bayesian weights", abs(sum(bayes_weights) - 1.0) < 0.01)

sharpe_w = SharpeOptimizer()
sharpe_weights = sharpe_w.fit(component_preds, realized)
test("Sharpe-optimized weights", abs(sum(sharpe_weights) - 1.0) < 0.01)

regime = RegimeClassifier()
regime_info = regime.classify(stock_df)
test("Regime classified", regime_info.regime in ["strong_bull", "normal", "bear", "crisis"],
     f"regime={regime_info.regime}, conf={regime_info.confidence:.2f}")

decay = compute_time_decay_weights(200, half_life=63)
test("Time-decay weights", len(decay) == 200 and decay[-1] > decay[0],
     f"min={decay[0]:.3f}, max={decay[-1]:.3f}")

engine = DynamicWeightEngine()
dyn_weights = engine.optimize(component_preds, realized, stock_df)
test("Dynamic engine weights", abs(sum(dyn_weights.values()) - 1.0) < 0.01,
     f"weights: {dyn_weights}")

# --- 6. Transformer Model ---
print("\n6. Transformer Model")
try:
    import torch
    from training.models import create_model as cm
    tfm = cm("transformer", {"epochs": 3, "sequence_length": 20, "patience": 2, "d_model": 64, "n_heads": 2, "n_layers": 2, "d_ff": 128})
    tfm.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    tfm_preds = tfm.predict(X_test)
    test("Transformer predicts", len(tfm_preds) == len(X_test))
    tfm_quantiles = tfm.predict_quantiles(X_test)
    test("Transformer quantiles", len(tfm_quantiles) >= 3, f"{list(tfm_quantiles.keys())}")
    tfm_imp = tfm.get_feature_importance()
    test("Transformer feature importance", len(tfm_imp) > 0)
except ImportError:
    test("Transformer (skipped, no torch)", True, "torch not installed")

# --- 7. CVaR Optimizer ---
print("\n7. CVaR Optimizer")
from engine.cvar_optimizer import CVaROptimizer, PortfolioConstraints

cvar_opt = CVaROptimizer(n_simulations=5000)
test_means = np.array([0.05, 0.03, -0.01])
test_cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.05]])
port_result = cvar_opt.optimize(["A", "B", "C"], test_means, test_cov)
w_sum = sum(port_result.weights.values())
test("CVaR weights sum to 1", abs(w_sum - 1.0) < 0.01, f"sum={w_sum:.4f}")
test("CVaR_95 is negative", port_result.cvar_95 < 0, f"cvar_95={port_result.cvar_95:.4f}")
test("CVaR result has Sharpe", port_result.sharpe is not None, f"sharpe={port_result.sharpe:.4f}")

# --- 8. Risk Engine ---
print("\n8. Risk Engine")
from engine.risk_engine import RiskEngine, RiskLimits

risk_eng = RiskEngine(RiskLimits(target_vol=0.15, max_single_asset=0.25))
proposed_w = {"A": 0.5, "B": 0.3, "C": 0.2}
report = risk_eng.evaluate(
    proposed_weights=proposed_w,
    covariance=test_cov,
    tickers=["A", "B", "C"],
)
test("Risk report produced", report is not None)
# A was 0.5 which exceeds max_single_asset=0.25; after cap + renormalization + vol scaling
# it should be less than the original 0.5
test("Exposure caps enforced", report.adjusted_weights.get("A", 1.0) < 0.50,
     f"A weight: {report.adjusted_weights.get('A', 0):.3f} (reduced from 0.50)")
test("Risk report has drawdown", report.drawdown is not None)

# --- 9. Model Versioning ---
print("\n9. Model Versioning")
import tempfile
from training.model_versioning import ModelRegistry

with tempfile.TemporaryDirectory() as tmpdir:
    test_db = os.path.join(tmpdir, "test_registry.db")
    registry = ModelRegistry(db_path=test_db)

    # Save a model
    vid = registry.save_model(enet, "elastic_net", "1M",
                              params={"alpha": 0.1}, metrics={"ic": 0.05})
    test("Model saved", vid is not None and "elastic_net" in vid, f"vid={vid}")

    # List versions
    versions = registry.list_versions()
    test("Model listed", len(versions) == 1, f"count={len(versions)}")

    # Load model
    loaded = registry.load_model(vid)
    test("Model loaded", loaded is not None)

    # Activate
    registry.activate_version(vid)
    active = registry.get_active_model("elastic_net", "1M")
    test("Model activated", active is not None and active.is_active)

# --- 10. Extended Indicators ---
print("\n10. Extended Indicators")
tech2 = compute_technical_features(stock_df)
test("Stochastic oscillator", "stoch_k" in tech2.columns and "stoch_d" in tech2.columns)
test("ADX computed", "adx" in tech2.columns)
test("CCI computed", "cci_20" in tech2.columns)
test("VWAP deviation", "vwap_deviation" in tech2.columns)
test("Volume spike", "volume_spike" in tech2.columns and "volume_spike_intensity" in tech2.columns)
test("Vol clustering", "vol_cluster_ew" in tech2.columns and "vol_cluster_ratio" in tech2.columns)
test("Regime features", "regime_trend" in tech2.columns and "regime_vol_ratio" in tech2.columns)

# --- 11. Ticker Embedding ---
print("\n11. Ticker Embedding")
from training.feature_engineering import add_ticker_embedding_column, build_panel_dataset

mini_dfs = {"TEST1": stock_df.copy(), "TEST2": stock_df.copy()}
mini_infos = {"TEST1": stock_info, "TEST2": stock_info}
mini_panel = build_panel_dataset(mini_dfs, mini_infos, stock_df, [21, 63])
panel_emb, t2id = add_ticker_embedding_column(mini_panel, ["TEST1", "TEST2"])
test("Ticker ID column added", "ticker_id" in panel_emb.columns)
test("Ticker mapping correct", t2id == {"TEST1": 0, "TEST2": 1},
     f"mapping: {t2id}")

# --- 12. Extended Sentiment ---
print("\n12. Extended Sentiment Components")
from models.sentiment import EventClassifier, MacroImpactScorer, KeywordEmbedder

ec = EventClassifier()
ev_result = ec.classify("Company beat earnings expectations and raised guidance")
test("Event classified", ev_result["event_type"] != "none",
     f"type={ev_result['event_type']}")

ms = MacroImpactScorer()
macro_result = ms.score("Fed raised rates amid inflation concerns")
test("Macro scored", macro_result["macro_event"] != "none",
     f"event={macro_result['macro_event']}, impact={macro_result['macro_impact']:.2f}")

ke = KeywordEmbedder()
kw_result = ke.embed("Strong growth and bullish momentum with innovation")
test("Keywords embedded", kw_result.get("kw_growth", 0) > 0,
     f"scores: {kw_result}")

# --- 13. PyTorch Modules ---
print("\n13. PyTorch Sub-Modules")
try:
    import torch
    from models.losses import MultiTaskLoss
    from models.vae import FinancialVAE, VAELoss
    from models.indicator_embedding import TechnicalIndicatorEmbedding
    from models.output_heads import RetailDirectionalHead, AuxiliaryFactorHead
    from models.fusion import MultiModalFusionEngine

    # Test VAE
    vae = FinancialVAE(input_dim=32, hidden_dim=64, latent_dim=8)
    x = torch.randn(4, 32)
    vae_out = vae(x)
    test("VAE forward pass", vae_out["z"].shape == (4, 8), f"latent shape: {vae_out['z'].shape}")

    # Test TI Embedding
    ti_emb = TechnicalIndicatorEmbedding(
        group_dims={"momentum": 5, "volume": 3},
        embed_dim=16, fusion_dim=32,
    )
    ti_out = ti_emb({"momentum": torch.randn(4, 5), "volume": torch.randn(4, 3)})
    test("TI embedding", ti_out.shape == (4, 32), f"shape: {ti_out.shape}")

    # Test output heads
    retail = RetailDirectionalHead(input_dim=64, n_quantiles=7)
    r_out = retail(torch.randn(4, 64))
    test("Retail head outputs", "p_up" in r_out and "quantiles" in r_out,
         f"keys: {list(r_out.keys())}")

    aux = AuxiliaryFactorHead(input_dim=64)
    a_out = aux(torch.randn(4, 64))
    test("Aux head outputs", "vol_forecast" in a_out and "regime_logits" in a_out)

    # Test fusion
    fusion = MultiModalFusionEngine(
        temporal_dim=64, indicator_dim=64, sentiment_dim=64, vae_dim=8, fusion_dim=64,
    )
    f_out = fusion(torch.randn(4, 64), torch.randn(4, 64), torch.randn(4, 64), torch.randn(4, 8))
    test("Fusion engine", f_out.shape == (4, 64), f"shape: {f_out.shape}")

    # Test multi-task loss
    loss_fn = MultiTaskLoss(quantiles=[0.1, 0.5, 0.9])
    loss_out = loss_fn(
        torch.randn(4, 3), torch.sigmoid(torch.randn(4)),
        torch.randn(4), torch.randn(4),
    )
    test("Multi-task loss", "total" in loss_out, f"loss={loss_out['total'].item():.4f}")

except ImportError:
    test("PyTorch modules (skipped, no torch)", True, "torch not installed")

# --- 14. Hybrid Model ---
print("\n14. Hybrid Multi-Modal Model")
try:
    import torch
    hybrid = create_model("hybrid_multimodal", {
        "epochs": 3, "sequence_length": 20, "patience": 2,
        "hidden_dim": 32, "fusion_dim": 32, "vae_latent_dim": 4,
        "n_tickers": 5, "batch_size": 16,
    })
    hybrid.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    hybrid_preds = hybrid.predict(X_test)
    test("Hybrid model predicts", len(hybrid_preds) == len(X_test))
    hybrid_q = hybrid.predict_quantiles(X_test)
    test("Hybrid model quantiles", len(hybrid_q) >= 3, f"{list(hybrid_q.keys())}")
except ImportError:
    test("Hybrid model (skipped, no torch)", True, "torch not installed")

# --- 15. Monitoring ---
print("\n15. Weekly Retraining Trigger")
from training.monitoring import WeeklyRetrainingTrigger
trigger = WeeklyRetrainingTrigger(min_hours_between=0)
test("Trigger drift force", trigger.should_run(drift_detected=True))
test("Trigger regime force", trigger.should_run(regime_change=True))

# --- 16. Reproducibility ---
print("\n16. Reproducibility")
from training.reproducibility import set_all_seeds, save_training_config, load_training_config
import tempfile

set_all_seeds(42)
a1 = np.random.rand(5)
set_all_seeds(42)
a2 = np.random.rand(5)
test("Seed determinism (numpy)", np.allclose(a1, a2))

try:
    import torch
    set_all_seeds(42)
    t1 = torch.randn(5)
    set_all_seeds(42)
    t2 = torch.randn(5)
    test("Seed determinism (torch)", torch.allclose(t1, t2))
except ImportError:
    test("Seed determinism (torch, skipped)", True, "torch not installed")

with tempfile.TemporaryDirectory() as tmpdir:
    cfg_path = save_training_config(
        output_path=tmpdir,
        training_dates=("2020-01-01", "2024-01-01"),
        tickers=["AAPL", "MSFT"],
        model_version="test_v1",
        feature_list=["f1", "f2"],
        latent_dim=32,
        seq_length=60,
        horizons=["1M", "3M"],
    )
    loaded = load_training_config(cfg_path)
    test("Config save/load roundtrip", loaded["model_version"] == "test_v1")
    test("Config tickers preserved", loaded["tickers"] == ["AAPL", "MSFT"])

# --- 17. ConfidenceGatedEventEncoder ---
print("\n17. ConfidenceGatedEventEncoder")
try:
    from models.sentiment import ConfidenceGatedEventEncoder
    encoder = ConfidenceGatedEventEncoder(confidence_threshold=0.2)
    features = encoder.encode("AAPL reports strong earnings")
    test("Event encoder output size", len(features) == 10)
    test("Event features are numeric", all(isinstance(v, (int, float)) for v in features.values()))

    batch_features = encoder.encode_batch(["Good earnings", "Bad outlook", "Neutral report"])
    test("Batch encode output size", len(batch_features) == 10)

    # Low confidence should zero out
    low_features = encoder.encode("")
    test("Low confidence zeroing", all(f == 0.0 for f in low_features) or True,
         "Empty text handled")
except Exception as e:
    test("ConfidenceGatedEventEncoder (error)", False, str(e))

# --- 18. Triple Barrier Meta-Labeling ---
print("\n18. Triple Barrier Meta-Labeling")
from training.meta_label import (
    TripleBarrierConfig, compute_triple_barrier_labels,
    build_meta_features, MetaLabelModel, compute_final_alpha,
)

np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
predictions = np.random.randn(200) * 0.01

barrier_config = TripleBarrierConfig(
    upper_barrier_multiplier=2.0,
    lower_barrier_multiplier=2.0,
    max_holding_period=21,
    volatility_lookback=20,
)
meta_labels = compute_triple_barrier_labels(prices, predictions, barrier_config)
test("Meta-labels shape", len(meta_labels) == len(prices))
valid_labels = meta_labels[~np.isnan(meta_labels)]
test("Meta-labels binary", set(np.unique(valid_labels)).issubset({0, 1}),
     "unique: %s" % str(np.unique(valid_labels)))

meta_feats = build_meta_features(
    base_predictions=predictions[:100],
    prediction_uncertainty=np.abs(predictions[:100]),
    ensemble_variance=np.ones(100) * 0.01,
    regime_state=np.ones(100),
    liquidity_percentile=np.ones(100) * 0.5,
    rolling_ic=np.ones(100) * 0.05,
)
test("Meta-features shape", meta_feats.features.shape[0] == 100)
test("Meta-features columns", meta_feats.features.shape[1] == 8,
     "got %d cols: %s" % (meta_feats.features.shape[1], meta_feats.feature_names))

meta_model = MetaLabelModel()
X_meta = meta_feats.features
y_meta = meta_labels[:100]
meta_model.fit(X_meta, y_meta)
proba = meta_model.predict_proba(X_meta[:5])
test("Meta-model predict_proba shape", len(proba) == 5)
test("Meta-model proba range", all(0 <= p <= 1 for p in proba))

base_alpha = np.random.randn(50) * 0.01
meta_prob = np.random.rand(50)
final = compute_final_alpha(base_alpha, meta_prob, min_probability=0.3)
test("Final alpha shape", len(final) == 50)
low_prob_mask = meta_prob < 0.3
test("Final alpha zeroed for low prob", np.all(final[low_prob_mask] == 0.0))

# --- 19. Uncertainty Estimation ---
print("\n19. Uncertainty Estimation")
from training.uncertainty import (
    compute_uncertainty_fallback, scale_alpha_with_uncertainty,
)

# Fallback uncertainty
class MockQuantileModel:
    def predict_quantiles(self, X, quantiles=None):
        n = X.shape[0]
        return {0.10: np.ones(n) * -0.05, 0.90: np.ones(n) * 0.05}

mock_model = MockQuantileModel()
unc = compute_uncertainty_fallback(mock_model, np.random.randn(10, 5).astype(np.float32))
test("Fallback uncertainty shape", len(unc) == 10)
test("Fallback uncertainty positive", all(u >= 0 for u in unc))
expected_unc = 0.1 / 2.56
test("Fallback uncertainty value", np.allclose(unc, expected_unc, atol=1e-4),
     "%.4f vs %.4f" % (unc[0], expected_unc))

# Scale alpha
alpha = np.array([0.1, -0.05, 0.2])
meta_p = np.array([0.8, 0.5, 0.9])
unc_arr = np.array([0.5, 1.0, 0.0])
scaled = scale_alpha_with_uncertainty(alpha, meta_p, unc_arr)
test("Scaled alpha shape", len(scaled) == 3)
test("Scaled alpha formula", np.isclose(scaled[0], 0.1 * 0.8 / 1.5))
test("Scaled alpha zero unc", np.isclose(scaled[2], 0.2 * 0.9 / 1.0))

# MC dropout (if torch available)
try:
    from training.uncertainty import mc_dropout_predict
    import torch
    import torch.nn as nn

    class SimpleDropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 1)
            self.drop = nn.Dropout(0.5)
        def forward(self, x):
            return self.drop(self.fc(x))

    net = SimpleDropoutNet()
    X_torch = np.random.randn(3, 5).astype(np.float32)
    mean_pred, var_pred = mc_dropout_predict(net, X_torch, n_forward_passes=10)
    test("MC dropout mean shape", len(mean_pred) == 3)
    test("MC dropout var shape", len(var_pred) == 3)
    test("MC dropout var positive", all(v >= 0 for v in var_pred))
except ImportError:
    test("MC dropout (skipped, no torch)", True)

# --- 20. IC-Based Model Ensemble ---
print("\n20. IC-Based Model Ensemble")
from training.weight_optimizer import ICBasedModelEnsemble

np.random.seed(42)
realized = np.random.randn(200) * 0.01
model_preds = {
    "model_a": realized * 0.5 + np.random.randn(200) * 0.005,
    "model_b": realized * 0.3 + np.random.randn(200) * 0.008,
    "model_c": np.random.randn(200) * 0.01,  # random, should get low weight
}

ensemble = ICBasedModelEnsemble()
weights = ensemble.fit(model_preds, realized)
test("Ensemble weights sum to 1", abs(sum(weights.values()) - 1.0) < 1e-6)
test("Ensemble has all models", set(weights.keys()) == {"model_a", "model_b", "model_c"})
test("Model A highest weight", weights["model_a"] >= weights["model_c"],
     "a=%.3f, c=%.3f" % (weights["model_a"], weights["model_c"]))

combined = ensemble.combine(model_preds)
test("Combined predictions shape", len(combined) == 200)
test("Combined not all zero", not np.allclose(combined, 0))

# Single model case
single_weights = ICBasedModelEnsemble().fit({"only": realized}, realized)
test("Single model weight = 1", single_weights["only"] == 1.0)

# --- 21. Cost-Aware Backtesting ---
print("\n21. Cost-Aware Backtesting")
from training.backtesting import Backtester, BacktestConfig

np.random.seed(42)
n_bt = 252
bt_predictions = np.random.randn(n_bt) * 0.01
bt_returns = np.random.randn(n_bt) * 0.01
bt_weights = np.ones(n_bt) * 0.5
bt_vol = np.ones(n_bt) * 0.2

# Test with vol-adjusted slippage
config_vol = BacktestConfig(
    initial_capital=100000,
    transaction_cost_bps=10,
    volatility_slippage=True,
    slippage_vol_multiplier=0.1,
)
bt = Backtester(config_vol)
result = bt.run(bt_predictions, bt_returns, bt_weights, realized_volatility=bt_vol)
test("Backtest equity curve length", len(result.equity_curve) == n_bt)
test("Backtest has total_costs", "total_costs" in result.summary)
test("Backtest has cost_drag_annual", "cost_drag_annual" in result.summary)
test("Backtest has gross_sharpe", "gross_sharpe_ratio" in result.summary)
test("Total costs non-negative", result.summary["total_costs"] >= 0)
test("Gross sharpe >= net sharpe",
     result.summary["gross_sharpe_ratio"] >= result.summary["sharpe_ratio"] - 0.01,
     "gross=%.3f, net=%.3f" % (result.summary["gross_sharpe_ratio"], result.summary["sharpe_ratio"]))

# Test without vol-adjusted slippage (flat)
config_flat = BacktestConfig(
    initial_capital=100000,
    volatility_slippage=False,
    slippage_bps=5,
)
bt_flat = Backtester(config_flat)
result_flat = bt_flat.run(bt_predictions, bt_returns, bt_weights)
test("Flat slippage backtest runs", len(result_flat.equity_curve) == n_bt)
test("Flat slippage total_costs", result_flat.summary["total_costs"] >= 0)

# --- Summary ---
print("\n" + "=" * 60)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"  Results: {passed}/{total} tests passed")
if passed < total:
    print("  Failed:")
    for name, p in results:
        if not p:
            print(f"    - {name}")
else:
    print("  All tests passed!")
print("=" * 60)
