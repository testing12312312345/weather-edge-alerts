"""
Morning Follow-up Alert

Runs at 7:05am PT (15:05 UTC).
Checks if yesterday's YES limit sells triggered.
Reports on tail settlement status.
"""

import json
import os
import re
from datetime import datetime, date, timedelta

import requests

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

CITIES = {
    "PHX": {"series": "KXHIGHTPHX", "name": "Phoenix"},
    "LAX": {"series": "KXHIGHLAX", "name": "Los Angeles"},
    "LV": {"series": "KXHIGHTLV", "name": "Las Vegas"},
}


def find_today_event(series_ticker: str) -> dict | None:
    """Find the event for today's weather date."""
    today = date.today()
    ticker_date = today.strftime("%y%b%d").upper()
    expected_ticker = f"{series_ticker}-{ticker_date}"

    resp = requests.get(f"{KALSHI_API}/events/{expected_ticker}", timeout=15)
    if resp.status_code == 200:
        return resp.json().get("event")

    resp = requests.get(
        f"{KALSHI_API}/events",
        params={"series_ticker": series_ticker, "status": "open", "limit": 10},
        timeout=15,
    )
    resp.raise_for_status()
    for event in resp.json().get("events", []):
        ticker = event["event_ticker"]
        m = re.search(r"(\d{2})([A-Z]{3})(\d{2})$", ticker)
        if not m:
            continue
        try:
            weather_date = datetime.strptime(
                f"20{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%b-%d"
            ).date()
            if weather_date == today:
                return event
        except ValueError:
            continue
    return None


def get_orderbook_price(ticker: str) -> float:
    """Get current mid price in cents."""
    resp = requests.get(f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=15)
    if resp.status_code != 200:
        return 0
    data = resp.json().get("orderbook_fp", {})
    yes_bids = data.get("yes_dollars", [])
    no_bids = data.get("no_dollars", [])
    best_yes_bid = max((float(b[0]) for b in yes_bids), default=0) if yes_bids else 0
    best_no_bid = max((float(b[0]) for b in no_bids), default=0) if no_bids else 0
    best_yes_bid_c = best_yes_bid * 100
    best_yes_ask_c = (1.0 - best_no_bid) * 100 if best_no_bid > 0 else 0
    if best_yes_bid_c and best_yes_ask_c:
        return (best_yes_bid_c + best_yes_ask_c) / 2
    return best_yes_bid_c or best_yes_ask_c


def parse_bucket_label(title: str) -> str:
    m = re.search(r"be\s+([<>≤≥]?\d+[-–]?\d*)[°\s]", title)
    if m:
        return m.group(1) + "°"
    m = re.search(r"(\d+)°?\s+or\s+(above|below)", title, re.I)
    if m:
        return f">{m.group(1)}°" if m.group(2).lower() == "above" else f"<{m.group(1)}°"
    return title


def main():
    print(f"Morning Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    today = date.today()
    lines = [
        f"# ☀️ Morning Update — {today.strftime('%A, %B %d')}",
        "",
    ]

    for city_key, city_info in CITIES.items():
        series = city_info["series"]
        name = city_info["name"]

        event = find_today_event(series)
        if not event:
            lines.append(f"## {name}: No event found")
            continue

        event_ticker = event["event_ticker"]

        resp = requests.get(
            f"{KALSHI_API}/markets",
            params={"event_ticker": event_ticker, "limit": 20},
            timeout=15,
        )
        markets = resp.json().get("markets", [])

        if not markets:
            lines.append(f"## {name}: No markets")
            continue

        lines.append(f"## {name}")

        buckets = []
        for mkt in markets:
            label = parse_bucket_label(mkt.get("title", ""))
            price = get_orderbook_price(mkt["ticker"])
            result = mkt.get("result", "")
            buckets.append({"label": label, "price": price, "result": result})

        buckets.sort(key=lambda b: b["price"], reverse=True)

        # Show current state
        for b in buckets:
            if b["result"] == "yes":
                lines.append(f"   ✅ **{b['label']}** — SETTLED WINNER")
            elif b["result"] == "no":
                lines.append(f"   ❌ {b['label']} — settled loser")
            else:
                bar = "█" * int(b["price"] / 5) if b["price"] > 0 else ""
                lines.append(f"   {b['label']}: {b['price']:.0f}¢ {bar}")

        # Favorite status
        fav = buckets[0] if buckets else None
        if fav and fav["result"] == "":
            if fav["price"] > 50:
                lines.append(f"   📈 Favorite **{fav['label']}** at {fav['price']:.0f}¢ — looking strong")
            elif fav["price"] > 30:
                lines.append(f"   ➡️ Favorite **{fav['label']}** at {fav['price']:.0f}¢ — holding")
            else:
                lines.append(f"   📉 Favorite **{fav['label']}** dropped to {fav['price']:.0f}¢")

        # Check tails
        cheap = [b for b in buckets if b["price"] < 10]
        if cheap:
            safe = all(b["price"] < 10 for b in cheap)
            if safe:
                lines.append(f"   🔴 Tails still cheap — NO bets look good")
            else:
                rising = [b for b in cheap if b["price"] > 5]
                if rising:
                    lines.append(f"   ⚠️ Tail **{rising[0]['label']}** rising to {rising[0]['price']:.0f}¢ — watch it")

        lines.append("")

    lines.append("*Settlement happens after market close. Tails resolve automatically.*")

    message = "\n".join(lines)

    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=15)
        print("Morning alert sent!")
    else:
        print(message)


if __name__ == "__main__":
    main()
