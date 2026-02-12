#!/usr/bin/env python3
"""
ATLAS Slack Bot — V8 Full-Spectrum Intelligence
Listens for @mentions and runs the ATLAS engine + V8 extended analysis,
pulling LIVE data from yfinance for any ticker.

Output is a 10-section institutional-grade research report:
verdict, fundamentals, valuation, technicals, peers, sentiment,
risk factors, growth catalysts, macro context, and engine signal.

Usage:
    python3 bot.py
"""

import os
import re
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from atlas_engine import run_atlas
from data_fetcher import fetch_live_data
from v8_data import fetch_v8_data
from v8_report import format_v8_report

# Load environment variables
load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
CAPITAL = int(os.environ.get("CAPITAL", "250000"))
FRED_API_KEY = os.environ.get("FRED_API_KEY")  # Optional, for better macro data
USE_LIVE_DATA = os.environ.get("USE_LIVE_DATA", "true").lower() == "true"
STATIC_DATA_PATH = os.environ.get("DATA_PATH", str(Path(__file__).parent / "data" / "default"))
STATE_DIR = str(Path(__file__).parent / "state")

if not SLACK_BOT_TOKEN:
    print("ERROR: SLACK_BOT_TOKEN not set. Copy .env.example to .env and fill in your tokens.")
    print("See SETUP.md for instructions.")
    exit(1)

if not SLACK_APP_TOKEN:
    print("ERROR: SLACK_APP_TOKEN not set. You need an app-level token (xapp-...) for Socket Mode.")
    print("See SETUP.md for instructions.")
    exit(1)

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)


def parse_mention(text):
    """
    Parse the mention text to extract a ticker symbol.
    Examples:
        "@atlas TSLA"   -> "TSLA"
        "@atlas AAPL"   -> "AAPL"
        "@atlas run"    -> "SPY" (default)
        "@atlas"        -> "SPY" (default)
    """
    # Remove the @mention tag
    cleaned = re.sub(r'<@[A-Z0-9]+>', '', text).strip().upper()

    # Look for a ticker-like word (1-5 uppercase letters)
    tokens = cleaned.split()
    for token in tokens:
        if re.match(r'^[A-Z]{1,5}$', token) and token not in ('RUN', 'HELP', 'STATUS', 'LIVE'):
            return token

    return 'SPY'  # Default to SPY


@app.event("app_mention")
def handle_atlas_mention(event, say, client):
    """
    Handle @atlas mentions in any channel.
    Fetches live data, runs ATLAS engine, posts trader-ready analysis.
    """
    channel = event['channel']
    user = event['user']
    text = event.get('text', '')
    thread_ts = event.get('ts')

    # Parse which symbol they want
    symbol = parse_mention(text)

    # Check for help command
    if 'HELP' in text.upper():
        say(
            text=(
                "*ATLAS Trading Engine* :chart_with_upwards_trend:\n"
                "Mention me with any ticker symbol to get analysis:\n"
                "  `@atlas AAPL` — Apple\n"
                "  `@atlas TSLA` — Tesla\n"
                "  `@atlas MSFT` — Microsoft\n"
                "  `@atlas SPY` — S&P 500 ETF\n"
                "  `@atlas` — Default (SPY)\n"
                "  `@atlas help` — This message\n\n"
                "Full report includes fundamentals, technicals, peers, news, macro, and more.\n"
                "Data is pulled live from Yahoo Finance."
            ),
            thread_ts=thread_ts
        )
        return

    # Acknowledge immediately
    say(
        text=f":gear: Running ATLAS V8 on *{symbol}*... pulling live data & building full report (30-45 sec)",
        thread_ts=thread_ts
    )

    try:
        # Fetch live data for ATLAS engine
        if USE_LIVE_DATA:
            print(f"[BOT] Fetching live data for {symbol}...")
            data_path = fetch_live_data(
                symbol=symbol,
                fred_api_key=FRED_API_KEY
            )
        else:
            data_path = STATIC_DATA_PATH

        # Run the 11-layer engine
        print(f"[BOT] Running ATLAS engine on {symbol}...")
        report_text, summary = run_atlas(
            symbol=symbol,
            data_path=data_path,
            capital=CAPITAL,
            state_dir=STATE_DIR
        )

        # Fetch V8 extended data (peers, technicals, news, macro, etc.)
        print(f"[BOT] Fetching V8 extended data for {symbol}...")
        v8_extended = fetch_v8_data(
            symbol=symbol,
            fred_api_key=FRED_API_KEY
        )

        # Generate V8 full-spectrum report (10 sections)
        print(f"[BOT] Formatting V8 report for {symbol}...")
        v8_messages = format_v8_report(summary, v8_extended)

        # Post each section as a threaded reply
        for msg in v8_messages:
            if isinstance(msg, dict):
                say(blocks=msg.get('blocks', []), text=msg.get('text', ''), thread_ts=thread_ts)
            else:
                say(text=msg, thread_ts=thread_ts)

        # Final confirmation
        say(
            text=f":white_check_mark: ATLAS V8 complete for *{symbol}* — {len(v8_messages)} sections delivered",
            thread_ts=thread_ts
        )

        # Clean up temp data directory
        if USE_LIVE_DATA and data_path != STATIC_DATA_PATH:
            import shutil
            try:
                shutil.rmtree(data_path)
            except:
                pass

    except ValueError as e:
        say(
            text=f":warning: *Data Error for {symbol}:* {str(e)}\n\nCheck that `{symbol}` is a valid US ticker symbol.",
            thread_ts=thread_ts
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[BOT] Error: {error_msg}")
        say(
            text=f":x: *ATLAS Error for {symbol}:*\n```\n{error_msg[-1500:]}\n```",
            thread_ts=thread_ts
        )


@app.event("message")
def handle_message_events(body, logger):
    """Catch-all for message events (required by slack-bolt to avoid warnings)."""
    pass


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    # Create state directory for meta-learning persistence
    os.makedirs(STATE_DIR, exist_ok=True)

    print("=" * 60)
    print("ATLAS Slack Bot — V8 Full-Spectrum Mode")
    print("=" * 60)
    print(f"Live data:  {'ENABLED (yfinance)' if USE_LIVE_DATA else 'DISABLED (static files)'}")
    print(f"FRED API:   {'Connected' if FRED_API_KEY else 'Not configured (using yfinance for macro)'}")
    print(f"Capital:    ${CAPITAL:,}")
    print(f"State dir:  {STATE_DIR}")
    print()

    if not USE_LIVE_DATA:
        data_dir = Path(STATIC_DATA_PATH)
        if data_dir.exists():
            files = list(data_dir.glob("*"))
            print(f"Static data: {len(files)} files in {STATIC_DATA_PATH}")
        else:
            print(f"WARNING: Static data not found: {STATIC_DATA_PATH}")

    print()
    print("Starting bot in Socket Mode...")
    print("Mention @atlas with any ticker in Slack.")
    print("Examples: @atlas AAPL, @atlas TSLA, @atlas SPY")
    print("Press Ctrl+C to stop.")
    print()

    # Health check server for Render (keeps the service alive)
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ATLAS is running")
        def log_message(self, format, *args):
            pass  # Suppress logs

    port = int(os.environ.get("PORT", 10000))
    health_server = HTTPServer(("0.0.0.0", port), HealthHandler)
    threading.Thread(target=health_server.serve_forever, daemon=True).start()
    print(f"Health check listening on port {port}")

    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
