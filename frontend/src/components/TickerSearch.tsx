"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";

export default function TickerSearch() {
  const [ticker, setTicker] = useState("");
  const router = useRouter();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const t = ticker.trim().toUpperCase();
    if (t) {
      router.push(`/symbol/${t}`);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        placeholder="Enter ticker (e.g. AAPL)"
        className="bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-4 py-2 text-sm
                   placeholder-gray-500 focus:outline-none focus:border-accent
                   w-64"
        aria-label="Stock ticker symbol"
      />
      <button
        type="submit"
        className="bg-accent text-gray-900 px-4 py-2 rounded-lg text-sm font-medium
                   hover:bg-cyan-300 transition-colors"
      >
        Analyze
      </button>
    </form>
  );
}
