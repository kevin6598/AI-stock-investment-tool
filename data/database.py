import sqlite3
import os
from datetime import datetime
from typing import List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            date_added TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_holding(ticker: str, shares: float, avg_cost: float, date_added: str = None):
    """Add a new holding. If ticker already exists, update shares and avg_cost."""
    if date_added is None:
        date_added = datetime.now().strftime("%Y-%m-%d")
    ticker = ticker.upper().strip()

    conn = get_connection()
    cursor = conn.cursor()

    # Check if holding exists
    cursor.execute("SELECT id, shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,))
    existing = cursor.fetchone()

    if existing:
        # Weighted average cost
        total_shares = existing["shares"] + shares
        new_avg_cost = (existing["shares"] * existing["avg_cost"] + shares * avg_cost) / total_shares
        cursor.execute(
            "UPDATE holdings SET shares = ?, avg_cost = ? WHERE id = ?",
            (total_shares, new_avg_cost, existing["id"]),
        )
    else:
        cursor.execute(
            "INSERT INTO holdings (ticker, shares, avg_cost, date_added) VALUES (?, ?, ?, ?)",
            (ticker, shares, avg_cost, date_added),
        )

    # Record transaction
    cursor.execute(
        "INSERT INTO transactions (ticker, action, shares, price, date) VALUES (?, ?, ?, ?, ?)",
        (ticker, "BUY", shares, avg_cost, date_added),
    )

    conn.commit()
    conn.close()


def delete_holding(holding_id: int):
    """Delete a holding by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get holding info for transaction record
    cursor.execute("SELECT ticker, shares, avg_cost FROM holdings WHERE id = ?", (holding_id,))
    holding = cursor.fetchone()
    if holding:
        cursor.execute(
            "INSERT INTO transactions (ticker, action, shares, price, date) VALUES (?, ?, ?, ?, ?)",
            (holding["ticker"], "SELL", holding["shares"], holding["avg_cost"],
             datetime.now().strftime("%Y-%m-%d")),
        )
        cursor.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))

    conn.commit()
    conn.close()


def get_all_holdings() -> List[dict]:
    """Return all holdings as list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, ticker, shares, avg_cost, date_added FROM holdings ORDER BY ticker")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_transactions(ticker: str = None) -> List[dict]:
    """Return transactions, optionally filtered by ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    if ticker:
        cursor.execute(
            "SELECT * FROM transactions WHERE ticker = ? ORDER BY date DESC", (ticker.upper(),)
        )
    else:
        cursor.execute("SELECT * FROM transactions ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
