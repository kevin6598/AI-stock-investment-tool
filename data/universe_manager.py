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

    @staticmethod
    def load_default_universe() -> List[TickerMembership]:
        """Load a default SP500-like universe with approximate date ranges.

        This is a representative subset. In production, load from a file
        with complete membership history.
        """
        members = [
            # Tech
            TickerMembership("AAPL", "2000-01-01", None, "Technology"),
            TickerMembership("MSFT", "2000-01-01", None, "Technology"),
            TickerMembership("GOOGL", "2006-01-01", None, "Technology"),
            TickerMembership("AMZN", "2005-07-01", None, "Consumer Discretionary"),
            TickerMembership("META", "2013-12-23", None, "Technology"),
            TickerMembership("NVDA", "2001-11-30", None, "Technology"),
            TickerMembership("TSLA", "2020-12-21", None, "Consumer Discretionary"),
            # Finance
            TickerMembership("JPM", "2000-01-01", None, "Financials"),
            TickerMembership("BAC", "2000-01-01", None, "Financials"),
            TickerMembership("GS", "2002-07-01", None, "Financials"),
            TickerMembership("WFC", "2000-01-01", None, "Financials"),
            # Healthcare
            TickerMembership("JNJ", "2000-01-01", None, "Healthcare"),
            TickerMembership("UNH", "2000-01-01", None, "Healthcare"),
            TickerMembership("PFE", "2000-01-01", None, "Healthcare"),
            TickerMembership("ABBV", "2013-01-02", None, "Healthcare"),
            # Consumer
            TickerMembership("KO", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("PG", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("WMT", "2000-01-01", None, "Consumer Staples"),
            TickerMembership("HD", "2000-01-01", None, "Consumer Discretionary"),
            # Industrial
            TickerMembership("BA", "2000-01-01", None, "Industrials"),
            TickerMembership("CAT", "2000-01-01", None, "Industrials"),
            TickerMembership("GE", "2000-01-01", "2018-06-26", "Industrials"),
            # Energy
            TickerMembership("XOM", "2000-01-01", None, "Energy"),
            TickerMembership("CVX", "2000-01-01", None, "Energy"),
            # Telecom / Utilities
            TickerMembership("VZ", "2000-01-01", None, "Communication Services"),
            TickerMembership("T", "2000-01-01", None, "Communication Services"),
            # Other notable
            TickerMembership("V", "2009-01-01", None, "Financials"),
            TickerMembership("MA", "2008-07-18", None, "Financials"),
            TickerMembership("DIS", "2000-01-01", None, "Communication Services"),
            TickerMembership("NFLX", "2010-12-20", None, "Communication Services"),
        ]
        return members

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
