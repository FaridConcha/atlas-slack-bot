# ATLAS Slack Bot — Setup Guide

Get the ATLAS trading engine running as a Slack bot on your Mac in ~10 minutes.

## Step 1: Create the Slack App

1. Go to **https://api.slack.com/apps** and click **Create New App** → **From scratch**
2. Name it **ATLAS** (or whatever you like), select your workspace
3. Under **Socket Mode** (left sidebar):
   - Toggle **Enable Socket Mode** ON
   - Create an app-level token with name "atlas-socket" and scope `connections:write`
   - **Copy the token** (starts with `xapp-...`) — you'll need this
4. Under **OAuth & Permissions** (left sidebar):
   - Scroll to **Scopes → Bot Token Scopes** and add:
     - `app_mentions:read`
     - `chat:write`
   - Click **Install to Workspace** at the top
   - **Copy the Bot User OAuth Token** (starts with `xoxb-...`)
5. Under **Event Subscriptions** (left sidebar):
   - Toggle **Enable Events** ON
   - Under **Subscribe to bot events**, add: `app_mention`
   - Click **Save Changes**

## Step 2: Install Dependencies

Open Terminal on your Mac and run:

```bash
cd ~/Downloads/atlas-slack-bot

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip3 install -r requirements.txt
```

## Step 3: Configure Tokens

```bash
# Copy the example env file
cp .env.example .env

# Edit with your tokens
open -e .env
```

Fill in the two tokens you copied in Step 1:
```
SLACK_BOT_TOKEN=xoxb-your-actual-token
SLACK_APP_TOKEN=xapp-your-actual-token
```

Save and close.

## Step 4: Run the Bot

```bash
# Make sure you're in the bot directory with venv active
cd ~/Downloads/atlas-slack-bot
source venv/bin/activate

python3 bot.py
```

You should see:
```
============================================================
ATLAS Slack Bot
============================================================
Data path:  ./data/ko_live
Capital:    $250,000
Data files: 7 found in ./data/ko_live
  - breadth.csv
  - consensus.json
  - fundamentals.json
  - global_overnight.json
  - ko_ohlcv.csv
  - macro_rates.csv
  - volatility.csv

Starting bot in Socket Mode...
Mention @atlas in any channel to run analysis.
Press Ctrl+C to stop.
```

## Step 5: Test in Slack

1. Go to any channel in your Slack workspace
2. Invite the bot: `/invite @ATLAS`
3. Type: `@ATLAS KO`
4. The bot will reply with a summary + full report in a thread

## Usage

| Command | What it does |
|---------|-------------|
| `@atlas KO` | Run ATLAS analysis on Coca-Cola |
| `@atlas` | Run on default ticker (KO) |
| `@atlas help` | Show available commands |

## Troubleshooting

**Bot doesn't respond:**
- Make sure `python3 bot.py` is running in your terminal
- Make sure you invited the bot to the channel (`/invite @ATLAS`)
- Check that Socket Mode is enabled in your Slack app settings

**"SLACK_BOT_TOKEN not set":**
- Make sure `.env` exists (not just `.env.example`)
- Make sure the tokens are filled in correctly

**"Data Error: Could not load price data":**
- Check that `data/ko_live/` contains the 7 CSV/JSON files
- The path in `.env` should point to the data directory

**Import errors:**
- Make sure you activated the virtual environment: `source venv/bin/activate`
- Make sure you installed requirements: `pip3 install -r requirements.txt`

## Architecture

```
@atlas KO (Slack mention)
    → Socket Mode receives event (no ngrok needed!)
    → bot.py parses mention, extracts ticker
    → atlas_engine.py runs 8 scoring engines
    → message_formatter.py splits report for Slack
    → Bot posts summary + report as threaded replies
```

## Files

| File | Purpose |
|------|---------|
| `bot.py` | Slack event handler — the main entry point |
| `atlas_engine.py` | The ATLAS V12+ trading engine (8 engines + regime + probabilistic framework + sizing) |
| `v8_data.py` | Full-spectrum data fetcher, DCF valuation with Monte Carlo simulations |
| `valuation_config.py` | Single source of truth for all institutional valuation constants |
| `web_server.py` | FastAPI web dashboard with interactive charts and MC panels |
| `test_data_integrity.py` | Regression test suite (238 tests) |
| `message_formatter.py` | Splits long reports into Slack-friendly chunks |
| `data/ko_live/` | KO market data (CSV + JSON files) |
| `.env` | Your Slack tokens (not committed to git) |
| `requirements.txt` | Python dependencies |
