"use client";

import { useState, useEffect, FormEvent } from "react";
import {
  api,
  RetailPrediction,
  ExposureGuidanceResponse,
} from "@/lib/api";
import { formatReturn, directionColor } from "@/lib/utils";

interface Holding {
  ticker: string;
  shares: number;
  avgPrice: number;
  prediction?: RetailPrediction;
}

const STORAGE_KEY = "portfolio_holdings";

function loadHoldings(): Holding[] {
  if (typeof window === "undefined") return [];
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch {}
  return [];
}

function saveHoldings(holdings: Holding[]) {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(holdings.map(({ ticker, shares, avgPrice }) => ({ ticker, shares, avgPrice })))
    );
  } catch {}
}

export default function PortfolioPage() {
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [exposure, setExposure] = useState<ExposureGuidanceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Form state
  const [newTicker, setNewTicker] = useState("");
  const [newShares, setNewShares] = useState("");
  const [newAvgPrice, setNewAvgPrice] = useState("");

  useEffect(() => {
    setMounted(true);
    const saved = loadHoldings();
    if (saved.length > 0) setHoldings(saved);
    api.getExposureGuidance().then(setExposure).catch(() => {});
  }, []);

  function addHolding(e: FormEvent) {
    e.preventDefault();
    const ticker = newTicker.trim().toUpperCase();
    const shares = parseFloat(newShares);
    const avgPrice = parseFloat(newAvgPrice) || 0;

    if (!ticker || isNaN(shares) || shares <= 0) return;

    // If ticker exists, update shares
    const existing = holdings.find((h) => h.ticker === ticker);
    let updated: Holding[];
    if (existing) {
      updated = holdings.map((h) =>
        h.ticker === ticker
          ? { ...h, shares: h.shares + shares, avgPrice: avgPrice || h.avgPrice }
          : h
      );
    } else {
      updated = [...holdings, { ticker, shares, avgPrice }];
    }

    setHoldings(updated);
    saveHoldings(updated);
    setNewTicker("");
    setNewShares("");
    setNewAvgPrice("");
    setAnalyzed(false);
  }

  function removeHolding(ticker: string) {
    const updated = holdings.filter((h) => h.ticker !== ticker);
    setHoldings(updated);
    saveHoldings(updated);
    setAnalyzed(false);
  }

  function clearAll() {
    setHoldings([]);
    saveHoldings([]);
    setAnalyzed(false);
  }

  async function analyzePortfolio() {
    if (holdings.length === 0) return;
    setLoading(true);
    const updated = [...holdings];

    for (let i = 0; i < updated.length; i++) {
      try {
        const res = await api.predict(updated[i].ticker, "3M");
        updated[i] = { ...updated[i], prediction: res.prediction };
      } catch {
        // Skip failed predictions
      }
    }

    setHoldings(updated);
    setAnalyzed(true);
    setLoading(false);
  }

  const mult = exposure?.exposure_multiplier ?? 1;
  const warningLevel = exposure?.warning_level || "HEALTHY";

  // Portfolio stats
  const analyzedHoldings = holdings.filter((h) => h.prediction);
  const totalConfidence = analyzedHoldings.length > 0
    ? analyzedHoldings.reduce((s, h) => s + (h.prediction?.confidence || 0), 0) / analyzedHoldings.length
    : 0;
  const avgReturn = analyzedHoldings.length > 0
    ? analyzedHoldings.reduce((s, h) => s + (h.prediction?.point_estimate || 0), 0) / analyzedHoldings.length
    : 0;
  const bullish = analyzedHoldings.filter((h) => h.prediction?.direction === "UP").length;
  const bearish = analyzedHoldings.filter((h) => h.prediction?.direction === "DOWN").length;

  if (!mounted) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Portfolio</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            보유 종목 관리 및 AI 분석 (early warning 노출 조정 적용)
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={analyzePortfolio}
            disabled={loading || holdings.length === 0}
            className="bg-accent text-gray-900 px-4 py-2 rounded-lg text-sm font-medium
                       hover:bg-cyan-300 transition-colors disabled:opacity-50"
          >
            {loading ? "분석 중..." : "AI 분석 실행"}
          </button>
        </div>
      </div>

      {/* Exposure context */}
      {exposure && (
        <div className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 mb-6 flex items-center justify-between">
          <div className="text-sm flex items-center gap-4">
            <span className="text-gray-500">전략 상태:</span>
            <span
              className={`font-bold ${
                warningLevel === "HEALTHY"
                  ? "text-emerald-400"
                  : warningLevel === "CAUTION"
                  ? "text-yellow-400"
                  : warningLevel === "WARNING"
                  ? "text-orange-400"
                  : "text-rose-400"
              }`}
            >
              {warningLevel}
            </span>
            <span className="text-gray-500">노출 배수:</span>
            <span className="font-mono font-bold text-accent">
              {(mult * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-xs text-gray-500">
            {exposure.recommended_action}
          </div>
        </div>
      )}

      {/* Add Holding Form */}
      <div className="bg-white dark:bg-card rounded-xl p-5 border border-gray-200 dark:border-gray-700 mb-6">
        <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
          종목 추가
        </h2>
        <form onSubmit={addHolding} className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="text-xs text-gray-500 block mb-1">티커</label>
            <input
              type="text"
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value)}
              placeholder="AAPL"
              className="bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                         placeholder-gray-500 focus:outline-none focus:border-accent w-28"
              required
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">수량</label>
            <input
              type="number"
              value={newShares}
              onChange={(e) => setNewShares(e.target.value)}
              placeholder="100"
              min="0.01"
              step="any"
              className="bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                         placeholder-gray-500 focus:outline-none focus:border-accent w-28"
              required
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">평균단가 ($)</label>
            <input
              type="number"
              value={newAvgPrice}
              onChange={(e) => setNewAvgPrice(e.target.value)}
              placeholder="150.00"
              min="0"
              step="any"
              className="bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                         placeholder-gray-500 focus:outline-none focus:border-accent w-32"
            />
          </div>
          <button
            type="submit"
            className="bg-accent text-gray-900 px-4 py-2 rounded-lg text-sm font-medium
                       hover:bg-cyan-300 transition-colors"
          >
            추가
          </button>
          {holdings.length > 0 && (
            <button
              type="button"
              onClick={clearAll}
              className="text-xs text-gray-500 hover:text-rose-400 transition-colors px-3 py-2"
            >
              전체 삭제
            </button>
          )}
        </form>
      </div>

      {/* Portfolio Stats */}
      {analyzed && analyzedHoldings.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard label="종목 수" value={`${holdings.length}`} />
          <StatCard
            label="평균 신뢰도"
            value={`${(totalConfidence * 100).toFixed(1)}%`}
            color="text-accent"
          />
          <StatCard
            label="평균 예상수익"
            value={formatReturn(avgReturn)}
            color={avgReturn >= 0 ? "text-up" : "text-down"}
          />
          <StatCard
            label="상승 / 하락"
            value={`${bullish} / ${bearish}`}
            color=""
            upDown
            up={bullish}
            down={bearish}
          />
        </div>
      )}

      {/* Holdings Table */}
      {holdings.length > 0 ? (
        <div className="bg-white dark:bg-card rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>티커</th>
                <th>수량</th>
                <th>평균단가</th>
                <th>AI 시그널</th>
                <th>예상 수익률</th>
                <th>신뢰도</th>
                <th>리스크</th>
                <th>노출 조정</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {holdings.map((h) => {
                const adjustedWeight = h.prediction
                  ? mult
                  : 1;
                return (
                  <tr key={h.ticker}>
                    <td>
                      <a
                        href={`/symbol/${h.ticker}`}
                        className="text-accent hover:underline font-medium"
                      >
                        {h.ticker}
                      </a>
                    </td>
                    <td className="font-mono">{h.shares}</td>
                    <td className="font-mono text-xs">
                      {h.avgPrice > 0 ? `$${h.avgPrice.toFixed(2)}` : "--"}
                    </td>
                    <td>
                      {h.prediction ? (
                        <span
                          className={`signal-badge ${
                            h.prediction.direction === "UP"
                              ? "signal-up"
                              : h.prediction.direction === "DOWN"
                              ? "signal-down"
                              : "signal-hold"
                          }`}
                        >
                          {h.prediction.direction}
                        </span>
                      ) : (
                        <span className="text-gray-500 text-xs">
                          {analyzed ? "실패" : "미분석"}
                        </span>
                      )}
                    </td>
                    <td>
                      {h.prediction ? (
                        <span
                          className={`font-mono text-xs ${directionColor(
                            h.prediction.direction
                          )}`}
                        >
                          {formatReturn(h.prediction.point_estimate)}
                        </span>
                      ) : (
                        <span className="text-gray-500">--</span>
                      )}
                    </td>
                    <td>
                      {h.prediction ? (
                        <span className="font-mono text-xs">
                          {(h.prediction.confidence * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-gray-500">--</span>
                      )}
                    </td>
                    <td>
                      {h.prediction ? (
                        <div className="w-16">
                          <div className="gauge-track">
                            <div
                              className="gauge-fill bg-rose-500"
                              style={{
                                width: `${h.prediction.risk_score * 100}%`,
                              }}
                            />
                          </div>
                          <div className="text-[10px] text-gray-500 mt-0.5 font-mono">
                            {(h.prediction.risk_score * 100).toFixed(0)}%
                          </div>
                        </div>
                      ) : (
                        <span className="text-gray-500">--</span>
                      )}
                    </td>
                    <td>
                      {h.prediction ? (
                        <span
                          className={`font-mono text-xs ${
                            adjustedWeight < 1 ? "text-yellow-400" : "text-emerald-400"
                          }`}
                        >
                          {(adjustedWeight * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="text-gray-500">--</span>
                      )}
                    </td>
                    <td>
                      <button
                        onClick={() => removeHolding(h.ticker)}
                        className="text-gray-500 hover:text-rose-400 transition-colors text-xs"
                        title="종목 삭제"
                      >
                        삭제
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-16 bg-white dark:bg-card rounded-xl border border-gray-200 dark:border-gray-700">
          <p className="text-lg text-gray-400 mb-2">보유 종목이 없습니다</p>
          <p className="text-sm text-gray-500">
            위 폼에서 티커와 수량을 입력하여 종목을 추가하세요
          </p>
        </div>
      )}

      {/* Exposure explanation */}
      {analyzed && exposure && mult < 1 && (
        <div className="mt-6 bg-yellow-400/5 border border-yellow-400/20 rounded-xl p-4">
          <div className="text-sm text-yellow-400 font-medium mb-1">
            Early Warning 노출 조정 활성화
          </div>
          <p className="text-xs text-gray-400">
            현재 전략 경고 수준이 {warningLevel}이므로, 전체 포지션의{" "}
            {((1 - mult) * 100).toFixed(0)}%를 현금으로 전환하는 것을 권장합니다.
            "노출 조정" 열은 각 종목에 적용되는 배수를 보여줍니다.
          </p>
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
  upDown,
  up,
  down,
}: {
  label: string;
  value: string;
  color?: string;
  upDown?: boolean;
  up?: number;
  down?: number;
}) {
  return (
    <div className="bg-white dark:bg-card rounded-xl p-4 border border-gray-200 dark:border-gray-700 text-center">
      <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
      {upDown ? (
        <div className="text-2xl font-bold">
          <span className="text-up">{up}</span>
          {" / "}
          <span className="text-down">{down}</span>
        </div>
      ) : (
        <div className={`text-2xl font-bold ${color || ""}`}>{value}</div>
      )}
    </div>
  );
}
