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
import shutil
import time
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
import gemini_qa

# Load environment variables
load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
CAPITAL = int(os.environ.get("CAPITAL", "250000"))
FRED_API_KEY = os.environ.get("FRED_API_KEY")  # Optional, for better macro data
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Optional, for AI follow-up Q&A
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

# Thread cache for AI Q&A follow-up
CACHE_TTL_HOURS = 4
CACHE_MAX_SIZE = 50
_thread_cache = {}  # {thread_ts: {symbol, summary, v8_extended, timestamp, conversation_history}}


def _cleanup_cache():
    """Remove expired entries and enforce size limit."""
    cutoff = time.time() - (CACHE_TTL_HOURS * 3600)
    expired = [ts for ts, data in _thread_cache.items() if data['timestamp'] < cutoff]
    for ts in expired:
        del _thread_cache[ts]

    # Evict oldest entries if over size limit
    while len(_thread_cache) > CACHE_MAX_SIZE:
        oldest_ts = min(_thread_cache, key=lambda ts: _thread_cache[ts]['timestamp'])
        del _thread_cache[oldest_ts]
        expired.append(oldest_ts)

    if expired:
        print(f"[CACHE] Cleaned {len(expired)} thread(s). Active: {len(_thread_cache)}")


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
                "Data is pulled live from Yahoo Finance.\n\n"
                "After a report, reply in the thread to ask follow-up questions (AI-powered)."
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

        # Cache data for AI follow-up Q&A
        if GROQ_API_KEY:
            _thread_cache[thread_ts] = {
                'symbol': symbol,
                'summary': summary,
                'v8_extended': v8_extended,
                'timestamp': time.time(),
                'conversation_history': [],
            }
            _cleanup_cache()

        # Post each section as a threaded reply
        for msg in v8_messages:
            if isinstance(msg, dict):
                say(blocks=msg.get('blocks', []), text=msg.get('text', ''), thread_ts=thread_ts)
            else:
                say(text=msg, thread_ts=thread_ts)

        # Final confirmation
        say(
            text=f":white_check_mark: ATLAS V8 complete for *{symbol}* — {len(v8_messages)} sections delivered"
                + ("\n:brain: _Reply in this thread to ask follow-up questions (AI-powered)_" if GROQ_API_KEY else ""),
            thread_ts=thread_ts
        )

        # Clean up temp data directory
        if USE_LIVE_DATA and data_path != STATIC_DATA_PATH:
            try:
                shutil.rmtree(data_path)
            except OSError:
                pass

    except ValueError as e:
        say(
            text=f":warning: *Data Error for {symbol}:* {str(e)[:300]}\n\nCheck that `{symbol}` is a valid US ticker symbol.",
            thread_ts=thread_ts
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[BOT] Error: {error_msg}")
        # Sanitize: show only the final exception line, not full paths/traceback
        error_summary = str(e)[:500]
        say(
            text=f":x: *ATLAS Error for {symbol}:* {error_summary}\n\n_Check bot logs for full details._",
            thread_ts=thread_ts
        )


@app.event("message")
def handle_message_events(event, say, client, logger):
    """
    Handle thread replies for AI follow-up Q&A.
    Also serves as catch-all for message events (required by slack-bolt).
    """
    # Filter: Ignore bot's own messages
    if event.get('bot_id') or event.get('subtype'):
        return

    # Filter: Only process thread replies
    thread_ts = event.get('thread_ts')
    if not thread_ts:
        return

    # Filter: Only process threads we have cached (ATLAS report threads)
    cache_entry = _thread_cache.get(thread_ts)
    if not cache_entry:
        return

    # Filter: Skip if Groq not configured
    if not GROQ_API_KEY:
        return

    # Extract the question
    question = event.get('text', '').strip()
    if not question:
        return

    # Remove any @mentions from the question text
    question = re.sub(r'<@[A-Z0-9]+>\s*', '', question).strip()
    if not question:
        return

    # Check for bare ticker symbol → redirect
    ticker_match = re.match(r'^[A-Z]{1,5}$', question.upper())
    if ticker_match and question.upper() not in ('A', 'I', 'OK', 'NO', 'YES', 'WHY', 'HOW', 'WHAT'):
        symbol = cache_entry['symbol']
        say(
            text=f"This thread covers *{symbol}*. To analyze *{question.upper()}*, mention `@atlas {question.upper()}` in any channel.",
            thread_ts=thread_ts
        )
        return

    channel = event['channel']
    symbol = cache_entry['symbol']

    # Post thinking indicator
    thinking = None
    try:
        thinking = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f":brain: Thinking about your question on *{symbol}*..."
        )
    except Exception:
        pass

    # Call Groq
    try:
        answer = gemini_qa.ask(
            question=question,
            symbol=symbol,
            summary=cache_entry['summary'],
            v8_extended=cache_entry['v8_extended'],
            conversation_history=cache_entry.get('conversation_history', []),
        )

        # Store in conversation history for multi-turn
        cache_entry['conversation_history'].append(('user', question))
        cache_entry['conversation_history'].append(('model', answer))

        # Cap conversation history at 12 entries (6 exchanges)
        if len(cache_entry['conversation_history']) > 12:
            cache_entry['conversation_history'] = cache_entry['conversation_history'][-12:]

    except Exception as e:
        logger.error(f"Groq Q&A error: {e}")
        answer = f":x: Sorry, I couldn't process that question. Error: {str(e)[:200]}"

    # Delete thinking message, post answer
    if thinking:
        try:
            client.chat_delete(channel=channel, ts=thinking['ts'])
        except Exception:
            pass

    say(text=answer, thread_ts=thread_ts)


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

    # Initialize Groq AI (optional)
    if GROQ_API_KEY:
        groq_ok = gemini_qa.init_groq(GROQ_API_KEY)
        print(f"Groq AI:    {'Connected (follow-up Q&A enabled)' if groq_ok else 'FAILED (Q&A disabled)'}")
    else:
        print("Groq AI:    Not configured (set GROQ_API_KEY for follow-up Q&A)")
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
