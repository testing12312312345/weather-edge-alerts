"""
Weather Edge Alert System v2 — Forecast-Informed

Scans for underpriced Kalshi weather buckets using GFS/NBM forecasts.
Sends Discord alerts with exact buy instructions and fair value limit sells.

Runs 3x/day after GFS model runs land on Open-Meteo:
  - 11:30am PT (18:30 UTC) — after 12z GFS
  - 5:30pm PT (00:30 UTC) — after 18z GFS
  - 11:30pm PT (06:30 UTC) — after 00z GFS
"""

import json
import os
import re
import sys
import time
from datetime import datetime, date, timedelta

import requests
from scipy import stats
import numpy as np

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
BANKROLL = float(os.environ.get("BANKROLL", "200"))
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
SPREAD = 2.0  # cents

CITIES = {
    "PHX": {
        "series": "KXHIGHTPHX",
        "name": "Phoenix",
        "lat": 33.4373,
        "lon": -112.0078,
        "tz": "America/Phoenix",
        "blend": {"gfs": 1, "nbm": 1},
        "url_base": "https://kalshi.com/markets/kxhightphx/phoenix-high-temperature-daily",
    },
    "LAX": {
        "series": "KXHIGHLAX",
        "name": "Los Angeles",
        "lat": 33.9381,
        "lon": -118.3889,
        "tz": "America/Los_Angeles",
        "blend": {"gfs": 1},
        "url_base": "https://kalshi.com/markets/kxhighlax/highest-temperature-in-los-angeles",
    },
    "LV": {
        "series": "KXHIGHTLV",
        "name": "Las Vegas",
        "lat": 36.0800,
        "lon": -115.1522,
        "tz": "America/Los_Angeles",
        "blend": {"gfs": 1, "nbm": 1},
        "url_base": "https://kalshi.com/markets/kxhightlv/las-vegas-high-temperature-daily",
    },
    "MIA": {
        "series": "KXHIGHMIA",
        "name": "Miami",
        "lat": 25.7906,
        "lon": -80.3164,
        "tz": "America/New_York",
        "blend": {"gfs": 1},
        "url_base": "https://kalshi.com/markets/kxhighmia/miami-high-temperature-daily",
    },
}

# Pre-computed calibration from 300+ days of backtest data
# Updated periodically as more data accumulates
CALIBRATION = {
    "PHX": {"gfs_bias": -1.7, "nbm_bias": -1.5, "sigma": 1.5},
    "LAX": {"gfs_bias": -0.3, "sigma": 1.5},
    "LV":  {"gfs_bias": -0.5, "nbm_bias": 0.0, "sigma": 1.0},
    "MIA": {"gfs_bias": -1.4, "sigma": 1.1},
}


def get_weather_dates():
    """Get tomorrow's weather date. Same-day trades disabled pending
    separate sigma calibration for same-day forecasts.
    """
    utc_now = datetime.utcnow()
    pt_now = utc_now - timedelta(hours=7)  # PDT
    tomorrow = pt_now.date() + timedelta(days=1)
    return [tomorrow]


def fetch_forecast(city_key, weather_date):
    """Fetch GFS/NBM forecast from Open-Meteo."""
    city = CITIES[city_key]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": city["tz"],
        "forecast_days": 3,
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  Open-Meteo error: {resp.status_code}")
        return None

    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    target = weather_date.strftime("%Y-%m-%d")
    for i, d in enumerate(dates):
        if d == target and temps[i] is not None:
            return {"gfs": temps[i]}

    return None


def fetch_forecast_multimodel(city_key, weather_date):
    """Fetch GFS + NBM from Open-Meteo Historical Forecast API."""
    city = CITIES[city_key]
    target = weather_date.strftime("%Y-%m-%d")

    url = "https://api.open-meteo.com/v1/forecast"
    models_param = "gfs_seamless"
    if "nbm" in city["blend"]:
        models_param = "gfs_seamless,ncep_nbm_conus"

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": city["tz"],
        "forecast_days": 3,
        "models": models_param,
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        # Fallback to standard endpoint
        return fetch_forecast(city_key, weather_date)

    data = resp.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])

    # Find model-specific keys
    result = {}
    for key in daily:
        if key == "time":
            continue
        for i, d in enumerate(dates):
            if d == target and daily[key][i] is not None:
                if "gfs" in key:
                    result["gfs"] = daily[key][i]
                elif "nbm" in key:
                    result["nbm"] = daily[key][i]
                elif "temperature" in key and "gfs" not in result:
                    result["gfs"] = daily[key][i]

    return result if result else None


def find_tomorrow_event(series_ticker, weather_date):
    """Find Kalshi event for tomorrow."""
    ticker_date = weather_date.strftime("%y%b%d").upper()
    expected = f"{series_ticker}-{ticker_date}"

    resp = requests.get(f"{KALSHI_API}/events/{expected}", timeout=15)
    if resp.status_code == 200:
        return resp.json().get("event")

    # Fallback: search
    resp = requests.get(
        f"{KALSHI_API}/events",
        params={"series_ticker": series_ticker, "status": "open", "limit": 10},
        timeout=15,
    )
    if resp.status_code != 200:
        return None

    for event in resp.json().get("events", []):
        ticker = event["event_ticker"]
        m = re.search(r"(\d{2})([A-Z]{3})(\d{2})$", ticker)
        if not m:
            continue
        try:
            wd = datetime.strptime(f"20{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%b-%d").date()
            if wd == weather_date:
                return event
        except ValueError:
            continue
    return None


def get_orderbook_prices(markets):
    """Get mid prices from orderbook for all markets."""
    prices = {}
    for mkt in markets:
        ticker = mkt["ticker"]
        title = mkt.get("title", "")

        # Parse bucket label
        label = title
        m = re.search(r"be\s+([<>≤≥]?\d+[-–]?\d*)[°\s]", title)
        if m:
            label = m.group(1) + "°"
        else:
            m = re.search(r"(\d+)°?\s+or\s+(above|below)", title, re.I)
            if m:
                label = f">{m.group(1)}°" if m.group(2).lower() == "above" else f"<{m.group(1)}°"

        # Get orderbook
        resp = requests.get(f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=15)
        if resp.status_code != 200:
            continue

        data = resp.json().get("orderbook_fp", {})
        yes_bids = data.get("yes_dollars", [])
        no_bids = data.get("no_dollars", [])

        best_yes_bid = max((float(b[0]) for b in yes_bids), default=0) if yes_bids else 0
        best_no_bid = max((float(b[0]) for b in no_bids), default=0) if no_bids else 0

        yes_bid_c = best_yes_bid * 100
        yes_ask_c = (1.0 - best_no_bid) * 100 if best_no_bid > 0 else 0

        mid = 0
        if yes_bid_c > 0 and yes_ask_c > 0:
            mid = (yes_bid_c + yes_ask_c) / 2
        elif yes_bid_c > 0:
            mid = yes_bid_c
        elif yes_ask_c > 0:
            mid = yes_ask_c

        # Parse range
        rng = None
        l = label.replace("°", "")
        m2 = re.match(r"<(\d+)", l)
        if m2:
            rng = (-999, float(m2.group(1)) - 0.5)
        m2 = re.match(r">(\d+)", l)
        if m2:
            rng = (float(m2.group(1)) + 0.5, 999)
        m2 = re.match(r"(\d+)[-–](\d+)", l)
        if m2:
            rng = (float(m2.group(1)) - 0.5, float(m2.group(2)) + 0.5)

        if rng:
            prices[label] = {
                "mid": mid,
                "ticker": ticker,
                "range": rng,
                "title": title,
            }

    return prices


def compute_model_probs(forecast, city_key, bucket_prices):
    """Compute calibrated probabilities for each bucket."""
    city = CITIES[city_key]
    cal = CALIBRATION[city_key]
    blend = city["blend"]

    # Bias-correct and blend
    corrected = []
    for model in blend:
        if model in forecast:
            bias_key = f"{model}_bias"
            bias = cal.get(bias_key, 0)
            corrected.append(forecast[model] - bias)

    if not corrected:
        return {}

    blended_temp = np.mean(corrected)
    sigma = cal["sigma"]

    # Compute probability for each bucket
    probs = {}
    for label, info in bucket_prices.items():
        lo, hi = info["range"]
        if lo < -900:
            lo = -np.inf
        if hi > 900:
            hi = np.inf
        p = stats.norm.cdf(hi, blended_temp, sigma) - stats.norm.cdf(lo, blended_temp, sigma)
        probs[label] = max(p, 0.001)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def find_trades(bucket_prices, model_probs, bankroll, event_ticker, url_base):
    """Find underpriced buckets and compute trades."""
    if not model_probs:
        return []

    total_price = sum(info["mid"] for info in bucket_prices.values())
    if total_price < 30:
        return []

    trades = []

    # Find all underpriced buckets (edge > 3%)
    for label, info in bucket_prices.items():
        mid = info["mid"]
        if mid < 5 or mid > 60:
            continue

        market_prob = mid / total_price
        model_prob = model_probs.get(label, 0)
        edge = model_prob - market_prob

        if edge <= 0.03:
            continue

        # Edge-weighted allocation: 2% for small edge, up to 6% for large
        alloc = min(0.06, max(0.02, 0.02 + edge * 0.4))
        entry_cents = mid + SPREAD / 2
        entry_price = entry_cents / 100
        contracts = max(1, int(bankroll * alloc / entry_price))

        # Fair value = model probability * 100 cents
        fair_value_cents = model_prob * 100

        if fair_value_cents <= entry_cents:
            continue

        potential_profit = contracts * (fair_value_cents / 100 - entry_price)
        risk = contracts * entry_price

        trades.append({
            "label": label,
            "ticker": info["ticker"],
            "entry_price": mid,
            "contracts": contracts,
            "cost": contracts * entry_price,
            "fair_value": fair_value_cents,
            "edge": edge,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "potential_profit": potential_profit,
            "risk": risk,
            "alloc": alloc,
            "url": f"{url_base}/{event_ticker.lower()}",
        })

    # Sort by edge (best first)
    trades.sort(key=lambda t: t["edge"], reverse=True)

    # Return top 2 (don't overload)
    return trades[:2]


def find_tail_trades(bucket_prices, winning_label, bankroll, event_ticker, url_base):
    """Find tail NO trades (cheapest buckets)."""
    sorted_buckets = sorted(bucket_prices.items(), key=lambda x: x[1]["mid"])
    tails = []

    for label, info in sorted_buckets[:2]:
        mid = info["mid"]
        if mid <= 0 or mid > 8:
            continue

        no_cost = (100 - mid + SPREAD / 2) / 100
        contracts = max(1, int(bankroll * 0.05 / no_cost))
        cost = contracts * no_cost
        win_amount = contracts * (mid - SPREAD / 2) / 100

        tails.append({
            "label": label,
            "ticker": info["ticker"],
            "yes_price": mid,
            "contracts": contracts,
            "cost": cost,
            "win_amount": win_amount,
            "url": f"{url_base}/{event_ticker.lower()}",
        })

    return tails


def format_discord(all_date_trades, bankroll, scan_label):
    """Format Discord alert message for multiple weather dates."""
    lines = [
        f"# Weather Edge Alert",
        f"**{scan_label}** | Bankroll: ${bankroll:.0f}",
        "",
    ]

    total_cost = 0
    has_trades = False

    for weather_date, city_trades in all_date_trades.items():
        day_name = weather_date.strftime("%A, %B %d")
        is_today = weather_date == (datetime.utcnow() - timedelta(hours=7)).date()
        date_label = f"{day_name} ({'today' if is_today else 'tomorrow'})"
        lines.append(f"## {date_label}")
        lines.append("")

        for city_key, data in city_trades.items():
            city_name = CITIES[city_key]["name"]
            yes_trades = data.get("yes", [])
            tail_trades = data.get("tails", [])
            forecast = data.get("forecast", {})

            if not yes_trades and not tail_trades:
                lines.append(f"### {city_name}")
                lines.append("No edge found")
                lines.append("")
                continue

            fc_str = ", ".join(f"{k}={v:.1f}F" for k, v in forecast.items())
            lines.append(f"### {city_name} (forecast: {fc_str})")

            for t in yes_trades:
                has_trades = True
                lines.append(
                    f"**BUY YES** {t['label']} — edge {t['edge']:.0%}"
                )
                lines.append(
                    f"   Buy {t['contracts']} contracts @ {t['entry_price']:.0f}c = **${t['cost']:.2f}**"
                )
                lines.append(
                    f"   Limit sell @ **{t['fair_value']:.0f}c** (model fair value)"
                )
                lines.append(
                    f"   Model: {t['model_prob']:.0%} | Market: {t['market_prob']:.0%}"
                )
                lines.append(f"   {t['url']}")
                lines.append("")
                total_cost += t["cost"]

            for t in tail_trades:
                has_trades = True
                lines.append(
                    f"**SELL NO** {t['label']} (priced at {t['yes_price']:.0f}c)"
                )
                lines.append(
                    f"   Buy {t['contracts']} NO @ ${t['cost']:.2f}"
                )
                lines.append(f"   {t['url']}")
                lines.append("")
                total_cost += t["cost"]

    if not has_trades:
        lines.append("**No trades** — no buckets underpriced >3%")

    lines.append("---")
    lines.append(f"Total cost: **${total_cost:.2f}**")

    return "\n".join(lines)


def send_discord(message):
    """Send to Discord webhook."""
    if not DISCORD_WEBHOOK:
        print("No DISCORD_WEBHOOK set, printing instead:")
        print(message)
        return

    # Split if > 2000 chars
    if len(message) <= 2000:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=15)
        resp.raise_for_status()
    else:
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
            resp = requests.post(DISCORD_WEBHOOK, json={"content": chunk}, timeout=15)
            resp.raise_for_status()

    print("Discord alert sent!")


def expected_gfs_run():
    """Return the expected GFS run date and hour for this scan time."""
    utc_now = datetime.utcnow()
    utc_hour = utc_now.hour
    # Cron fires ~6h after model init: 18:30 UTC→12z, 00:30→18z, 06:30→00z
    if 16 <= utc_hour < 22:
        return utc_now.strftime("%Y%m%d"), "12"
    elif 22 <= utc_hour:
        return utc_now.strftime("%Y%m%d"), "18"
    elif utc_hour < 4:
        # 00:30 UTC → 18z run from previous day
        yesterday = utc_now - timedelta(days=1)
        return yesterday.strftime("%Y%m%d"), "18"
    else:
        # 06:30 UTC → 00z run from today
        return utc_now.strftime("%Y%m%d"), "00"


def check_noaa_gfs_available(run_date, run_hour):
    """Check if NOAA has published a specific GFS run."""
    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{run_date}/{run_hour}/atmos/"
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def check_forecast_freshness(max_retries=6):
    """Verify the expected GFS run has been published by NOAA before
    recommending trades.

    Our cron fires 6.5h after GFS model init. NOAA typically publishes
    ~5h after init, and Open-Meteo ingests within ~30-60 min of NOAA.
    So by the time we run, Open-Meteo has had the data for ~1 hour.

    The NOAA check is the authoritative gate: if the run directory exists,
    the data is available and Open-Meteo will have ingested it by now.

    Retries up to max_retries (5 min apart). Returns False if the
    expected run hasn't been published — caller should skip the scan.
    """
    run_date, run_hour = expected_gfs_run()
    print(f"  Expected GFS run: {run_hour}z on {run_date}")

    for attempt in range(max_retries):
        noaa_ready = check_noaa_gfs_available(run_date, run_hour)
        print(f"  NOAA {run_hour}z: {'AVAILABLE' if noaa_ready else 'NOT YET'}")

        if noaa_ready:
            print("  GFS run confirmed on NOAA — proceeding")
            return True

        if attempt < max_retries - 1:
            print(f"  Waiting 5 min ({attempt + 1}/{max_retries})...")
            time.sleep(300)

    print("  WARNING: GFS run not published on NOAA after 30 min — skipping scan")
    return False


def scan_date(weather_date):
    """Scan a single weather date across all cities. Returns dict of city trades."""
    city_trades = {}

    for city_key, city_info in CITIES.items():
        print(f"  Processing {city_info['name']} for {weather_date}...")

        # Fetch forecast
        forecast = fetch_forecast_multimodel(city_key, weather_date)
        if not forecast:
            print(f"    No forecast available")
            city_trades[city_key] = {"yes": [], "tails": [], "forecast": {}}
            continue

        print(f"    Forecast: {forecast}")

        # Find event
        event = find_tomorrow_event(city_info["series"], weather_date)
        if not event:
            print(f"    No event found for {weather_date}")
            city_trades[city_key] = {"yes": [], "tails": [], "forecast": forecast}
            continue

        event_ticker = event["event_ticker"]
        print(f"    Event: {event_ticker}")

        # Get markets and prices
        resp = requests.get(
            f"{KALSHI_API}/markets",
            params={"event_ticker": event_ticker, "limit": 20},
            timeout=15,
        )
        markets = resp.json().get("markets", [])
        print(f"    Markets: {len(markets)}")

        if not markets:
            city_trades[city_key] = {"yes": [], "tails": [], "forecast": forecast}
            continue

        bucket_prices = get_orderbook_prices(markets)
        bucket_str = ", ".join(f"{l}={info['mid']:.0f}c" for l, info in sorted(bucket_prices.items(), key=lambda x: x[1]["mid"], reverse=True))
        print(f"    Buckets: {bucket_str}")

        # Compute model probabilities
        model_probs = compute_model_probs(forecast, city_key, bucket_prices)
        if model_probs:
            prob_str = ", ".join(f"{l}={p:.0%}" for l, p in sorted(model_probs.items(), key=lambda x: x[1], reverse=True))
            print(f"    Model:   {prob_str}")

        # Find YES trades
        yes_trades = find_trades(bucket_prices, model_probs, BANKROLL, event_ticker, city_info["url_base"])
        for t in yes_trades:
            print(f"    TRADE: BUY {t['label']} @{t['entry_price']:.0f}c, limit sell @{t['fair_value']:.0f}c, edge={t['edge']:.0%}, {t['contracts']}c=${t['cost']:.2f}")

        # Find tail NO trades
        tail_trades = find_tail_trades(bucket_prices, None, BANKROLL, event_ticker, city_info["url_base"])

        city_trades[city_key] = {"yes": yes_trades, "tails": tail_trades, "forecast": forecast}

    return city_trades


def main():
    utc_now = datetime.utcnow()
    pt_hour = (utc_now.hour - 7) % 24

    # Determine scan label (cron fires at 11:30am, 5:30pm, 11:30pm PT)
    if pt_hour >= 10 and pt_hour < 16:
        scan_label = "12z GFS Scan (11:30am PT)"
    elif pt_hour >= 16 and pt_hour < 22:
        scan_label = "18z GFS Scan (5:30pm PT)"
    else:
        scan_label = "00z GFS Scan (11:30pm PT)"

    weather_dates = get_weather_dates()

    print(f"Weather Edge Alert v2 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scan: {scan_label}")
    print(f"Weather dates: {', '.join(str(d) for d in weather_dates)}")
    print(f"Bankroll: ${BANKROLL:.0f}")
    print()

    # Verify we have a fresh GFS run before scanning
    is_fresh = check_forecast_freshness()
    print()

    if not is_fresh:
        stale_msg = (
            f"# Weather Edge Alert — SKIPPED\n"
            f"**{scan_label}** | Bankroll: ${BANKROLL:.0f}\n\n"
            f"Could not confirm fresh GFS run after 30 min of retries. "
            f"Skipping this scan to avoid trading on stale data.\n"
            f"Next scan will retry."
        )
        send_discord(stale_msg)
        print("Scan skipped — stale forecast data")
        return

    all_date_trades = {}
    for weather_date in weather_dates:
        print(f"--- Scanning {weather_date} ---")
        all_date_trades[weather_date] = scan_date(weather_date)

    # Format and send
    message = format_discord(all_date_trades, BANKROLL, scan_label)
    send_discord(message)


if __name__ == "__main__":
    main()
