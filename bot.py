#!/usr/bin/env python3
"""
ATLAS Slack Bot — V9 Capital Allocation Intelligence
Listens for @mentions and runs the ATLAS engine + V9 owner intelligence layer,
pulling LIVE data from yfinance for any ticker.

V9 adds Buffett-aligned business owner assessment on top of V8 quant signals.
Output is an 11-section report: owner assessment, verdict, fundamentals,
valuation, technicals, peers, sentiment, risk, catalysts, macro, engine signal.

Usage:
    python3 bot.py
"""

import os
import sys
import re
import shutil
import signal
import time
import threading
import traceback
from pathlib import Path

# Force flush on every print so Render logs appear in real time
import functools
print = functools.partial(print, flush=True)  # noqa: A001

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from atlas_engine import run_atlas
from data_fetcher import fetch_live_data
from v8_data import fetch_v8_data
from v8_report import format_v8_report
from web_report import generate_and_store_report
import gemini_qa

# Load environment variables (don't override existing — Render sets them via dashboard)
load_dotenv(override=False)

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

# Cold-start / shutdown state
_boot_time = time.time()
_boot_complete = False
_COLD_START_WINDOW = 120  # seconds — generous for Render container spin-up
_last_channel = None
_last_thread_ts = None


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
    # Remove the @mention tag (use [^>]+ to handle any user-ID format Slack sends)
    cleaned = re.sub(r'<@[^>]+>', '', text).strip().upper()

    print(f"[PARSE] raw text: {repr(text)}")
    print(f"[PARSE] cleaned:  {repr(cleaned)}")

    # Look for a ticker-like word (1-5 uppercase letters)
    tokens = cleaned.split()
    print(f"[PARSE] tokens:   {tokens}")
    for token in tokens:
        if re.match(r'^[A-Z]{1,5}$', token) and token not in ('RUN', 'HELP', 'STATUS', 'LIVE'):
            print(f"[PARSE] matched ticker: {token}")
            return token

    print("[PARSE] no ticker matched, defaulting to SPY")
    return 'SPY'  # Default to SPY


def _is_cold_start():
    """Return True once if the first mention arrives within the cold-start window."""
    global _boot_complete
    if _boot_complete:
        return False
    _boot_complete = True
    return (time.time() - _boot_time) < _COLD_START_WINDOW


def _send_boot_progress(client, channel, thread_ts, symbol):
    """Post and update a boot-progress message in-place. Returns the message ts."""
    stages = [
        ":zzz: ATLAS is waking up from sleep... hang tight",
        ":satellite: Connecting data systems...",
        ":rocket: Systems online, fetching *{symbol}*...",
    ]
    # Post initial message
    resp = client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                   text=stages[0])
    msg_ts = resp["ts"]
    # Walk through remaining stages
    for stage in stages[1:]:
        time.sleep(1.5)
        try:
            client.chat_update(channel=channel, ts=msg_ts,
                               text=stage.replace("{symbol}", symbol))
        except Exception:
            pass
    return msg_ts


def _handle_shutdown(signum, frame):
    """Notify Slack before Render kills the process."""
    if _last_channel:
        try:
            app.client.chat_postMessage(
                channel=_last_channel,
                thread_ts=_last_thread_ts,
                text=":zzz: ATLAS is going offline (Render free-tier sleep). "
                     "Mention me again to wake up — takes ~30 seconds.",
            )
        except Exception:
            pass
    sys.exit(0)


@app.event("app_mention")
def handle_atlas_mention(event, say, client):
    """
    Handle @atlas mentions in any channel.
    Fetches live data, runs ATLAS engine, posts trader-ready analysis.
    """
    global _last_channel, _last_thread_ts

    channel = event['channel']
    user = event['user']
    text = event.get('text', '')
    thread_ts = event.get('ts')

    _last_channel = channel
    _last_thread_ts = thread_ts

    # Parse which symbol they want
    symbol = parse_mention(text)

    # Check for help command
    if 'HELP' in text.upper():
        say(
            text=(
                "*ATLAS V9 — Capital Allocation Intelligence* :chart_with_upwards_trend:\n"
                "Mention me with any ticker symbol to get analysis:\n"
                "  `@atlas AAPL` — Apple\n"
                "  `@atlas TSLA` — Tesla\n"
                "  `@atlas MSFT` — Microsoft\n"
                "  `@atlas SPY` — S&P 500 ETF\n"
                "  `@atlas` — Default (SPY)\n"
                "  `@atlas help` — This message\n\n"
                "V9 report: Owner assessment (business quality, moat, margin of safety) + "
                "full quant analysis (fundamentals, technicals, peers, news, macro).\n"
                "Data is pulled live from Yahoo Finance.\n\n"
                "After a report, reply in the thread to ask follow-up questions (AI-powered)."
            ),
            thread_ts=thread_ts
        )
        return

    # Detect cold start (first mention after Render spin-up)
    cold = _is_cold_start()

    if cold:
        progress_ts = _send_boot_progress(client, channel, thread_ts, symbol)
    else:
        # Acknowledge immediately (warm path)
        say(
            text=f":gear: Running ATLAS V9 on *{symbol}*... pulling live data & building owner assessment + full report (30-45 sec)",
            thread_ts=thread_ts
        )
        progress_ts = None

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

        if cold and progress_ts:
            try:
                client.chat_update(channel=channel, ts=progress_ts,
                                   text=f":chart_with_upwards_trend: Live data fetched, running V9 engine on *{symbol}*...")
            except Exception:
                pass

        # Run the 11-layer engine
        print(f"[BOT] Running ATLAS engine on {symbol}...")
        report_text, summary = run_atlas(
            symbol=symbol,
            data_path=data_path,
            capital=CAPITAL,
            state_dir=STATE_DIR
        )

        if cold and progress_ts:
            try:
                client.chat_update(channel=channel, ts=progress_ts,
                                   text=":brain: Engine complete, building owner assessment...")
            except Exception:
                pass

        # Fetch V8 extended data (peers, technicals, news, macro, etc.)
        print(f"[BOT] Fetching V8 extended data for {symbol}...")
        v8_extended = fetch_v8_data(
            symbol=symbol,
            fred_api_key=FRED_API_KEY
        )

        # Generate V8 full-spectrum report (10 sections)
        print(f"[BOT] Formatting V8 report for {symbol}...")
        v8_messages = format_v8_report(summary, v8_extended)

        if cold and progress_ts:
            try:
                client.chat_update(channel=channel, ts=progress_ts,
                                   text=":white_check_mark: Report ready, delivering...")
            except Exception:
                pass

        # Generate V9 narrative interpretation (LLM-powered, non-blocking)
        v9_narrative = None
        v9_scores = v8_extended.get('v9_scores', {}) if v8_extended else {}
        if GROQ_API_KEY and v9_scores.get('v9_decision'):
            try:
                print(f"[BOT] Generating V9 narrative for {symbol}...")
                v9_narrative = gemini_qa.generate_v9_narrative(v9_scores, summary, v8_extended)
                if v9_narrative:
                    # Store narrative in v8_extended for web dashboard persistence
                    v9_scores['v9_narrative'] = v9_narrative
                    # Insert narrative as second message (after Owner Assessment)
                    narrative_msg = f":classical_building: *V9 NARRATIVE ASSESSMENT — {symbol}*\n\n{v9_narrative}"
                    v8_messages.insert(1, narrative_msg)
            except Exception as e:
                print(f"[BOT] V9 narrative generation failed (non-fatal): {e}")

        # Generate web report (non-blocking — errors must not break Slack flow)
        report_url = None
        try:
            company_name = (v8_extended.get('company') or {}).get('name')
            provenance = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_confidence": summary.get("data_confidence"),
                "fallback_mode": "LIVE" if USE_LIVE_DATA else "STATIC_FALLBACK",
            }
            web_result = generate_and_store_report(
                symbol=symbol,
                company_name=company_name,
                thread_ts=thread_ts,
                summary=summary,
                v8_extended=v8_extended,
                provenance=provenance,
            )
            report_url = web_result["report_url"]
            print(f"[BOT] Web report ready: {report_url}")
        except Exception as e:
            print(f"[BOT] Web report generation failed (non-fatal): {e}")

        # Inject web report link into first and last Slack messages
        if report_url and v8_messages:
            header_link = f":mag: *Full Report* (charts + tables + full calc breakdown): <{report_url}|Open dashboard>\n\n"
            footer_link = f"\n\n:arrow_upper_right: More detail: <{report_url}|Full web report>"
            # Prepend link to first message
            first = v8_messages[0]
            if isinstance(first, str):
                v8_messages[0] = header_link + first
            elif isinstance(first, dict):
                v8_messages[0]['text'] = header_link + first.get('text', '')
            # Append link to last message
            last = v8_messages[-1]
            if isinstance(last, str):
                v8_messages[-1] = last + footer_link
            elif isinstance(last, dict):
                v8_messages[-1]['text'] = last.get('text', '') + footer_link

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
            text=f":white_check_mark: ATLAS V9 complete for *{symbol}* — {len(v8_messages)} sections delivered"
                + (f"\n:mag: <{report_url}|View full web report>" if report_url else "")
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
    global _last_channel, _last_thread_ts

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

    _last_channel = event['channel']
    _last_thread_ts = thread_ts

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
    print("ATLAS Slack Bot — V9 Capital Allocation Intelligence")
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

    # FastAPI web server for reports + health check (replaces basic HTTP health server)
    import uvicorn
    from web_server import app as web_app

    port = int(os.environ.get("PORT", 10000))
    uvicorn_config = uvicorn.Config(web_app, host="0.0.0.0", port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    threading.Thread(target=uvicorn_server.run, daemon=True).start()
    print(f"Web server (reports + health) listening on port {port}")

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
