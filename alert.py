"""
Weather Edge Alert System

Runs at 10:55pm PT (6:55 UTC) every night.
Fetches live Kalshi prices for PHX, LA, LV temperature markets.
Computes the straddle bets and sends Discord alert with exact instructions.
"""

import json
import os
import re
import sys
from datetime import datetime, date, timedelta

import requests

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
BANKROLL = float(os.environ.get("BANKROLL", "30"))
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

CITIES = {
    "PHX": {
        "series": "KXHIGHTPHX",
        "name": "Phoenix",
        "url_base": "https://kalshi.com/markets/kxhightphx/phoenix-high-temperature-daily",
    },
    "LAX": {
        "series": "KXHIGHLAX",
        "name": "Los Angeles",
        "url_base": "https://kalshi.com/markets/kxhighlax/highest-temperature-in-los-angeles",
    },
    "LV": {
        "series": "KXHIGHTLV",
        "name": "Las Vegas",
        "url_base": "https://kalshi.com/markets/kxhightlv/las-vegas-high-temperature-daily",
    },
}

TAIL_ALLOC = 0.05      # 5% per tail NO
YES_ALLOC = 0.03       # 3% for YES favorite
MAX_TAIL_PRICE = 8     # cents
YES_MIN_PRICE = 15     # cents
YES_MAX_PRICE = 50     # cents
TP_PCT = 0.30          # 30% take profit on YES


def find_tomorrow_event(series_ticker: str) -> dict | None:
    """Find the event for tomorrow's weather date."""
    # Use Pacific time since we run at ~11pm PT
    utc_now = datetime.utcnow()
    pt_now = utc_now - timedelta(hours=7)  # PDT = UTC-7 (March-November)
    tomorrow = pt_now.date() + timedelta(days=1)

    # Try to find event by constructing the ticker
    # Format: KXHIGHTPHX-26MAR31
    ticker_date = tomorrow.strftime("%y%b%d").upper()
    expected_ticker = f"{series_ticker}-{ticker_date}"

    # First try direct lookup
    resp = requests.get(f"{KALSHI_API}/events/{expected_ticker}", timeout=15)
    if resp.status_code == 200:
        return resp.json().get("event")

    # Fallback: search active events
    resp = requests.get(
        f"{KALSHI_API}/events",
        params={"series_ticker": series_ticker, "status": "open", "limit": 10},
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json().get("events", [])

    for event in events:
        ticker = event["event_ticker"]
        m = re.search(r"(\d{2})([A-Z]{3})(\d{2})$", ticker)
        if not m:
            continue
        try:
            weather_date = datetime.strptime(
                f"20{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%b-%d"
            ).date()
            if weather_date == tomorrow:
                return event
        except ValueError:
            continue

    return None


def get_markets(event_ticker: str) -> list[dict]:
    """Get all markets (buckets) for an event."""
    resp = requests.get(
        f"{KALSHI_API}/markets",
        params={"event_ticker": event_ticker, "limit": 20},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("markets", [])


def get_orderbook_price(ticker: str) -> dict:
    """Get best bid/ask from orderbook."""
    resp = requests.get(f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=15)
    if resp.status_code != 200:
        return {"yes_bid": 0, "yes_ask": 0, "mid": 0}

    data = resp.json().get("orderbook_fp", {})
    yes_bids = data.get("yes_dollars", [])
    no_bids = data.get("no_dollars", [])

    # Format: [["0.05", "100"], ...] = [price_dollars, quantity]
    best_yes_bid = max((float(b[0]) for b in yes_bids), default=0) if yes_bids else 0
    best_no_bid = max((float(b[0]) for b in no_bids), default=0) if no_bids else 0

    # Convert to cents
    best_yes_bid_c = best_yes_bid * 100
    best_yes_ask_c = (1.0 - best_no_bid) * 100 if best_no_bid > 0 else 0

    mid = 0
    if best_yes_bid_c > 0 and best_yes_ask_c > 0:
        mid = (best_yes_bid_c + best_yes_ask_c) / 2
    elif best_yes_bid_c > 0:
        mid = best_yes_bid_c
    elif best_yes_ask_c > 0:
        mid = best_yes_ask_c

    return {"yes_bid": best_yes_bid_c, "yes_ask": best_yes_ask_c, "mid": mid}


def parse_bucket_label(title: str) -> str:
    """Extract bucket label from market title."""
    m = re.search(r"be\s+([<>≤≥]?\d+[-–]?\d*)[°\s]", title)
    if m:
        return m.group(1) + "°"
    # Try "XX or above" / "XX or below" patterns
    m = re.search(r"(\d+)°?\s+or\s+(above|below)", title, re.I)
    if m:
        if m.group(2).lower() == "above":
            return f">{m.group(1)}°"
        return f"<{m.group(1)}°"
    return title


def compute_bets(markets: list[dict], bankroll: float, url_base: str, event_ticker: str = "") -> list[dict]:
    """Compute the straddle bets for a city."""
    # Get prices for all buckets
    buckets = []
    for mkt in markets:
        ticker = mkt["ticker"]
        title = mkt.get("title", "")
        label = parse_bucket_label(title)
        prices = get_orderbook_price(ticker)

        buckets.append({
            "ticker": ticker,
            "label": label,
            "title": title,
            "mid": prices["mid"],
            "yes_bid": prices["yes_bid"],
            "yes_ask": prices["yes_ask"],
            "url": f"{url_base}/{event_ticker.lower()}",
        })

    if len(buckets) < 4:
        return []

    # Sort by mid price
    buckets.sort(key=lambda b: b["mid"])

    bets = []

    # TAIL NOs: 2 cheapest under max_tail_price
    for b in buckets[:2]:
        if b["mid"] <= 0 or b["mid"] > MAX_TAIL_PRICE:
            continue

        no_cost = (100 - b["mid"]) / 100.0
        contracts = max(1, int(bankroll * TAIL_ALLOC / no_cost))
        cost = contracts * no_cost

        bets.append({
            "type": "NO",
            "label": b["label"],
            "ticker": b["ticker"],
            "yes_price": b["mid"],
            "no_cost": no_cost,
            "contracts": contracts,
            "cost": cost,
            "win_amount": contracts * (b["mid"] / 100.0),
            "lose_amount": cost,
            "exit": "Hold to settlement",
            "url": b["url"],
        })

    # YES FAVORITE: most expensive bucket in range
    for b in reversed(buckets):
        if YES_MIN_PRICE <= b["mid"] <= YES_MAX_PRICE:
            entry_price = b["mid"] / 100.0
            contracts = max(1, int(bankroll * YES_ALLOC / entry_price))
            cost = contracts * entry_price
            tp_price = b["mid"] * (1 + TP_PCT)

            bets.append({
                "type": "YES",
                "label": b["label"],
                "ticker": b["ticker"],
                "yes_price": b["mid"],
                "entry_price": entry_price,
                "contracts": contracts,
                "cost": cost,
                "tp_price": tp_price,
                "win_amount": contracts * TP_PCT * entry_price,
                "lose_amount": cost,
                "exit": f"Limit sell at {tp_price:.0f}¢ (+30%)",
                "url": b["url"],
            })
            break

    return bets


def format_discord_message(all_bets: dict, bankroll: float) -> str:
    """Format the Discord alert message."""
    utc_now = datetime.utcnow()
    pt_now = utc_now - timedelta(hours=8)
    tomorrow = pt_now.date() + timedelta(days=1)
    day_name = tomorrow.strftime("%A, %B %d")

    lines = [
        f"# 🌡️ Weather Edge — {day_name}",
        f"**Bankroll: ${bankroll:.2f}**",
        "",
    ]

    total_cost = 0
    total_potential_win = 0

    for city_key, city_data in all_bets.items():
        city_name = city_data["name"]
        bets = city_data["bets"]

        if not bets:
            lines.append(f"## {city_name}")
            lines.append("⚠️ No bets found (market not open or no valid prices)")
            lines.append("")
            continue

        lines.append(f"## {city_name}")

        for bet in bets:
            if bet["type"] == "NO":
                emoji = "🔴"
                lines.append(
                    f"{emoji} **NO** on **{bet['label']}** (priced at {bet['yes_price']:.0f}¢)"
                )
                lines.append(
                    f"   Buy {bet['contracts']} NO @ ${bet['no_cost']:.2f} = **${bet['cost']:.2f}**"
                )
                lines.append(f"   Win: +${bet['win_amount']:.2f} | Lose: -${bet['lose_amount']:.2f}")
                lines.append(f"   Exit: {bet['exit']}")
            else:
                emoji = "🟢"
                lines.append(
                    f"{emoji} **YES** on **{bet['label']}** (priced at {bet['yes_price']:.0f}¢)"
                )
                lines.append(
                    f"   Buy {bet['contracts']} YES @ ${bet['entry_price']:.2f} = **${bet['cost']:.2f}**"
                )
                lines.append(
                    f"   Win (+30%): +${bet['win_amount']:.2f} | Lose: -${bet['lose_amount']:.2f}"
                )
                lines.append(f"   Exit: **{bet['exit']}**")

            lines.append(f"   🔗 {bet['url']}")
            lines.append("")

            total_cost += bet["cost"]
            total_potential_win += bet["win_amount"]

        lines.append("")

    lines.append("---")
    lines.append(f"💰 **Total cost tonight: ${total_cost:.2f}**")
    lines.append(f"📈 Potential win (all hit): +${total_potential_win:.2f}")
    lines.append("")
    lines.append("*Tail NOs: hold to settlement. YES: set limit sell at +30%.*")

    return "\n".join(lines)


def send_discord(message: str):
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        print("No DISCORD_WEBHOOK set, printing instead:")
        print(message)
        return

    # Discord has 2000 char limit, split if needed
    if len(message) <= 2000:
        resp = requests.post(
            DISCORD_WEBHOOK,
            json={"content": message},
            timeout=15,
        )
        resp.raise_for_status()
    else:
        # Split by city sections
        chunks = []
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > 1900 and current:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)

        for chunk in chunks:
            resp = requests.post(
                DISCORD_WEBHOOK,
                json={"content": chunk},
                timeout=15,
            )
            resp.raise_for_status()

    print("Discord alert sent!")


def main():
    print(f"Weather Edge Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Bankroll: ${BANKROLL:.2f}")
    print(f"Cities: {', '.join(CITIES.keys())}")
    print()

    all_bets = {}

    for city_key, city_info in CITIES.items():
        series = city_info["series"]
        name = city_info["name"]

        print(f"Processing {name}...")

        event = find_tomorrow_event(series)
        if not event:
            print(f"  No event found for tomorrow")
            all_bets[city_key] = {"name": name, "bets": []}
            continue

        event_ticker = event["event_ticker"]
        print(f"  Event: {event_ticker}")

        markets = get_markets(event_ticker)
        print(f"  Markets: {len(markets)} buckets")

        if not markets:
            all_bets[city_key] = {"name": name, "bets": []}
            continue

        bets = compute_bets(markets, BANKROLL, city_info["url_base"], event_ticker)
        print(f"  Bets: {len(bets)}")
        for bet in bets:
            print(f"    {bet['type']} {bet['label']}: {bet['contracts']} contracts @ ${bet.get('entry_price', bet.get('no_cost', 0)):.2f}")

        all_bets[city_key] = {"name": name, "bets": bets}

    # Format and send
    message = format_discord_message(all_bets, BANKROLL)
    send_discord(message)


if __name__ == "__main__":
    main()
