"""
Universe Manager
-----------------
Manages stock universe membership with date-aware ticker lists.

Handles survivorship bias by tracking when stocks enter/leave the universe.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TickerMembership:
    """Membership record for a single ticker."""
    ticker: str
    start_date: str        # when the stock entered the universe (YYYY-MM-DD)
    end_date: Optional[str]  # None = still active; date = delisted/removed
    sector: Optional[str] = None


class UniverseManager:
    """Manages a stock universe with date-aware membership.

    Prevents survivorship bias by tracking which tickers were active
    at any given point in time.
    """

    def __init__(self, membership_file: Optional[str] = None):
        """
        Args:
            membership_file: Path to JSON file with membership records.
                If None, loads default universe.
        """
        if membership_file is not None:
            self._members = self._load_from_file(membership_file)
        else:
            self._members = self.load_default_universe()

    def get_active_tickers(self, as_of_date: str) -> List[str]:
        """Return tickers that were active on a given date.

        Args:
            as_of_date: Date string (YYYY-MM-DD).

        Returns:
            List of active ticker symbols.
        """
        active = []
        for m in self._members:
            if m.start_date <= as_of_date:
                if m.end_date is None or m.end_date >= as_of_date:
                    active.append(m.ticker)
        return sorted(active)

    def get_all_tickers_in_range(self, start: str, end: str) -> List[str]:
        """Return all tickers that were active at any point in the range.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            List of ticker symbols.
        """
        tickers = set()
        for m in self._members:
            # Ticker was active if its membership period overlaps [start, end]
            member_start = m.start_date
            member_end = m.end_date or "9999-12-31"
            if member_start <= end and member_end >= start:
                tickers.add(m.ticker)
        return sorted(tickers)

    def get_sector(self, ticker: str) -> Optional[str]:
        """Get sector for a ticker."""
        for m in self._members:
            if m.ticker == ticker:
                return m.sector
        return None

    def get_all_members(self) -> List[TickerMembership]:
        """Return all membership records."""
        return list(self._members)

    def get_market_ticker(self, tickers: List[str]) -> str:
        """Return appropriate market index based on ticker composition.

        Korean-majority -> ^KS11 (KOSPI), otherwise -> SPY.
        """
        korean = sum(1 for t in tickers if t.upper().endswith((".KS", ".KQ")))
        if korean > len(tickers) / 2:
            return "^KS11"
        return "SPY"

    def get_universe_by_market(self, market: str) -> List[str]:
        """Filter active tickers by market/exchange.

        Args:
            market: "US" for NYSE/NASDAQ, "KR" for KOSPI/KOSDAQ, "all" for everything.

        Returns:
            List of ticker symbols.
        """
        result = []
        for m in self._members:
            if m.end_date is not None:
                continue  # skip delisted
            ticker_upper = m.ticker.upper()
            if market == "US":
                if not ticker_upper.endswith((".KS", ".KQ")):
                    result.append(m.ticker)
            elif market == "KR":
                if ticker_upper.endswith((".KS", ".KQ")):
                    result.append(m.ticker)
            else:
                result.append(m.ticker)
        return sorted(result)

    @staticmethod
    def load_default_universe() -> List[TickerMembership]:
        """Load default universe: ~85 US + 15 Korean = ~100 tickers.

        This is a representative S&P 500 subset. In production, load from a file
        with complete membership history.
        """
        members = [
            # --- US Tech (15) ---
            TickerMembership("AAPL", "2000-01-01", None, "Technology"),
            TickerMembership("MSFT", "2000-01-01", None, "Technology"),
            TickerMembership("GOOGL", "2006-01-01", None, "Technology"),
            TickerMembership("AMZN", "2005-07-01", None, "Consumer Discretionary"),
            TickerMembership("META", "2013-12-23", None, "Technology"),
            TickerMembership("NVDA", "2001-11-30", None, "Technology"),
            TickerMembership("TSLA", "2020-12-21", None, "Consumer Discretionary"),
            TickerMembership("AVGO", "2009-08-06", None, "Technology"),
            TickerMembership("ORCL", "2000-01-01", None, "Technology"),
            TickerMembership("CRM", "2008-09-15", None, "Technology"),
            TickerMembership("AMD", "2017-03-20", None, "Technology"),
            TickerMembership("ADBE", "2003-11-19", None, "Technology"),
            TickerMembership("INTC", "2000-01-01", None, "Technology"),
            TickerMembership("CSCO", "2000-01-01", None, "Technology"),
            TickerMembership("QCOM", "2000-01-01", None, "Technology"),
            # --- US Finance (12) ---
            TickerMembership("JPM", "2000-01-01", None, "Financials"),
            TickerMembership("BAC", "2000-01-01", None, "Financials"),
            TickerMembership("GS", "2002-07-01", None, "Financials"),
            TickerMembership("WFC", "2000-01-01", None, "Financials"),
            TickerMembership("V", "2009-01-01", None, "Financials"),
            TickerMembership("MA", "2008-07-18", None, "Financials"),
            TickerMembership("MS", "2002-05-31", None, "Financials"),
            TickerMembership("BLK", "2011-04-04", None, "Financials"),
            TickerMembership("C", "2000-01-01", None, "Financials"),
            TickerMembership("SCHW", "2000-01-01", None, "Financials"),
            TickerMembership("AXP", "2000-01-01", None, "Financials"),
            TickerMembership("USB", "2000-01-01", None, "Financials"),
            # --- US Healthcare (12) ---
            TickerMembership("JNJ", "2000-01-01", None, "Healthcare"),
            TickerMembership("UNH", "2000-01-01", None, "Healthcare"),
            TickerMembership("PFE", "2000-01-01", None, "Healthcare"),
            TickerMembership("ABBV", "2013-01-02", None, "Healthcare"),
            TickerMembership("LLY", "2000-01-01", None, "Healthcare"),
            TickerMembership("MRK", "2000-01-01", None, "Healthcare"),
            TickerMembership("TMO", "2000-01-01", None, "Healthcare"),
            TickerMembership("ABT", "2000-01-01", None, "Healthcare"),
            TickerMembership("BMY", "2000-01-01", None, "Healthcare"),
            TickerMembership("AMGN", "2000-01-01", None, "Healthcare"),
            TickerMembership("GILD", "2004-07-01", None, "Healthcare"),
            TickerMembership("MDT", "2000-01-01", None, "Healthcare"),
            # --- US Consumer (12) ---
            TickerMembership("KO", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("PG", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("WMT", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("HD", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("PEP", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("COST", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("NKE", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("MCD", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("SBUX", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("TGT", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("LOW", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("CL", "2000-01-01", None, "Consumer Staples"),
            # --- US Industrial (10) ---
            TickerMembership("BA", "2000-01-01", None, "Industrials"),
            TickerMembership("CAT", "2000-01-01", None, "Industrials"),
            TickerMembership("GE", "2000-01-01", "2018-06-26", "Industrials"),
            TickerMembership("HON", "2000-01-01", None, "Industrials"),
            TickerMembership("UNP", "2000-01-01", None, "Industrials"),
            TickerMembership("UPS", "2002-03-14", None, "Industrials"),
            TickerMembership("RTX", "2000-01-01", None, "Industrials"),
            TickerMembership("DE", "2000-01-01", None, "Industrials"),
            TickerMembership("MMM", "2000-01-01", None, "Industrials"),
            TickerMembership("LMT", "2000-01-01", None, "Industrials"),
            # --- US Energy (6) ---
            TickerMembership("XOM", "2000-01-01", None, "Energy"),
            TickerMembership("CVX", "2000-01-01", None, "Energy"),
            TickerMembership("COP", "2000-01-01", None, "Energy"),
            TickerMembership("SLB", "2000-01-01", None, "Energy"),
            TickerMembership("EOG", "2000-01-01", None, "Energy"),
            TickerMembership("PSX", "2012-05-01", None, "Energy"),
            # --- US Communication / Utilities (8) ---
            TickerMembership("VZ", "2000-01-01", None, "Communication Services"),
            TickerMembership("T", "2000-01-01", None, "Communication Services"),
            TickerMembership("DIS", "2000-01-01", None, "Communication Services"),
            TickerMembership("NFLX", "2010-12-20", None, "Communication Services"),
            TickerMembership("CMCSA", "2000-01-01", None, "Communication Services"),
            TickerMembership("TMUS", "2013-05-01", None, "Communication Services"),
            # --- US Materials / Real Estate (8) ---
            TickerMembership("LIN", "2018-10-31", None, "Materials"),
            TickerMembership("APD", "2000-01-01", None, "Materials"),
            TickerMembership("SHW", "2000-01-01", None, "Materials"),
            TickerMembership("FCX", "2000-01-01", None, "Materials"),
            TickerMembership("NEM", "2000-01-01", None, "Materials"),
            TickerMembership("PLD", "2006-01-01", None, "Real Estate"),
            TickerMembership("AMT", "2007-11-19", None, "Real Estate"),
            TickerMembership("EQIX", "2015-10-09", None, "Real Estate"),
            # --- Korean KOSPI (.KS) (15) ---
            TickerMembership("005930.KS", "2000-01-01", None, "Semiconductors"),    # Samsung Electronics
            TickerMembership("000660.KS", "2000-01-01", None, "Semiconductors"),    # SK Hynix
            TickerMembership("373220.KS", "2021-01-15", None, "Automotive"),        # LG Energy Solution
            TickerMembership("207940.KS", "2017-04-19", None, "Biotechnology"),     # Samsung Biologics
            TickerMembership("005380.KS", "2000-01-01", None, "Automotive"),        # Hyundai Motor
            TickerMembership("006400.KS", "2000-01-01", None, "Technology"),        # Samsung SDI
            TickerMembership("051910.KS", "2000-01-01", None, "Consumer Staples"),  # LG Chem
            TickerMembership("003670.KS", "2000-01-01", None, "Technology"),        # POSCO Holdings
            TickerMembership("035420.KS", "2008-11-28", None, "Technology"),        # NAVER
            TickerMembership("035720.KS", "2010-06-10", None, "Technology"),        # Kakao
            TickerMembership("105560.KS", "2015-07-24", None, "Financials"),        # KB Financial
            TickerMembership("055550.KS", "2001-09-10", None, "Financials"),        # Shinhan Financial
            TickerMembership("000270.KS", "2000-01-01", None, "Automotive"),        # Kia
            TickerMembership("068270.KS", "2005-06-24", None, "Biotechnology"),     # Celltrion
            TickerMembership("028260.KS", "2000-01-01", None, "Industrials"),       # Samsung C&T
        ]
        return members

    @staticmethod
    def load_extended_universe() -> List[TickerMembership]:
        """Load extended universe with ~300 S&P 500 components + Korean stocks.

        For training on larger datasets. Use with --extended-universe flag.
        """
        base = UniverseManager.load_default_universe()
        extended_us = [
            # Additional Tech
            TickerMembership("PYPL", "2015-07-20", None, "Technology"),
            TickerMembership("INTU", "2000-12-05", None, "Technology"),
            TickerMembership("NOW", "2019-11-21", None, "Technology"),
            TickerMembership("AMAT", "2000-01-01", None, "Technology"),
            TickerMembership("MU", "2000-01-01", None, "Technology"),
            TickerMembership("LRCX", "2012-06-29", None, "Technology"),
            TickerMembership("KLAC", "2000-01-01", None, "Technology"),
            TickerMembership("SNPS", "2017-09-18", None, "Technology"),
            TickerMembership("CDNS", "2017-09-18", None, "Technology"),
            TickerMembership("MRVL", "2021-06-28", None, "Technology"),
            TickerMembership("FTNT", "2021-12-20", None, "Technology"),
            TickerMembership("PANW", "2019-06-24", None, "Technology"),
            # Additional Finance
            TickerMembership("PNC", "2000-01-01", None, "Financials"),
            TickerMembership("TFC", "2000-01-01", None, "Financials"),
            TickerMembership("CME", "2002-12-06", None, "Financials"),
            TickerMembership("ICE", "2013-11-13", None, "Financials"),
            TickerMembership("AON", "2000-01-01", None, "Financials"),
            TickerMembership("MMC", "2000-01-01", None, "Financials"),
            TickerMembership("CB", "2010-07-01", None, "Financials"),
            TickerMembership("SPGI", "2000-01-01", None, "Financials"),
            # Additional Healthcare
            TickerMembership("DHR", "2000-01-01", None, "Healthcare"),
            TickerMembership("SYK", "2000-01-01", None, "Healthcare"),
            TickerMembership("ISRG", "2009-06-12", None, "Healthcare"),
            TickerMembership("VRTX", "2013-09-23", None, "Healthcare"),
            TickerMembership("REGN", "2013-05-01", None, "Healthcare"),
            TickerMembership("ZTS", "2013-06-21", None, "Healthcare"),
            TickerMembership("BDX", "2000-01-01", None, "Healthcare"),
            TickerMembership("EW", "2011-06-28", None, "Healthcare"),
            # Additional Consumer
            TickerMembership("BKNG", "2009-11-06", None, "Consumer Discretionary"),
            TickerMembership("TJX", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("MAR", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("ORLY", "2015-06-22", None, "Consumer Discretionary"),
            TickerMembership("AZO", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("ROST", "2000-01-01", None, "Consumer Discretionary"),
            TickerMembership("EL", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("MNST", "2012-06-28", None, "Consumer Staples"),
            TickerMembership("KDP", "2018-07-09", None, "Consumer Staples"),
            TickerMembership("GIS", "2000-01-01", None, "Consumer Staples"),
            # Additional Industrials
            TickerMembership("GD", "2000-01-01", None, "Industrials"),
            TickerMembership("NOC", "2000-01-01", None, "Industrials"),
            TickerMembership("WM", "2000-01-01", None, "Industrials"),
            TickerMembership("CSX", "2000-01-01", None, "Industrials"),
            TickerMembership("NSC", "2000-01-01", None, "Industrials"),
            TickerMembership("EMR", "2000-01-01", None, "Industrials"),
            TickerMembership("ITW", "2000-01-01", None, "Industrials"),
            TickerMembership("FDX", "2000-01-01", None, "Industrials"),
            # Additional Energy
            TickerMembership("MPC", "2013-12-23", None, "Energy"),
            TickerMembership("VLO", "2000-01-01", None, "Energy"),
            TickerMembership("OXY", "2000-01-01", None, "Energy"),
            TickerMembership("KMI", "2012-05-25", None, "Energy"),
            TickerMembership("WMB", "2000-01-01", None, "Energy"),
            TickerMembership("HES", "2000-01-01", None, "Energy"),
            # Additional Communication / Utilities
            TickerMembership("CHTR", "2016-09-08", None, "Communication Services"),
            TickerMembership("NEE", "2000-01-01", None, "Utilities"),
            TickerMembership("DUK", "2000-01-01", None, "Utilities"),
            TickerMembership("SO", "2000-01-01", None, "Utilities"),
            TickerMembership("D", "2000-01-01", None, "Utilities"),
            TickerMembership("AEP", "2000-01-01", None, "Utilities"),
            TickerMembership("EXC", "2000-01-01", None, "Utilities"),
            TickerMembership("SRE", "2000-01-01", None, "Utilities"),
        ]
        return base + extended_us

    @staticmethod
    def _load_from_file(filepath: str) -> List[TickerMembership]:
        """Load membership records from a JSON file."""
        import json
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            members = []
            for entry in data:
                members.append(TickerMembership(
                    ticker=entry["ticker"],
                    start_date=entry["start_date"],
                    end_date=entry.get("end_date"),
                    sector=entry.get("sector"),
                ))
            return members
        except Exception as e:
            logger.error("Failed to load universe file %s: %s", filepath, e)
            return UniverseManager.load_default_universe()
