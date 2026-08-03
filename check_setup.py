"""
Standalone diagnostic: run this BEFORE running the Streamlit app to confirm
your Groq key and .env file are set up correctly.

Usage:
    python check_setup.py
"""

import os
import sys

# Check for a stale env var in the OS/shell BEFORE config.py loads .env.
# This is the #1 cause of ".env edits seem to be ignored" -- if GROQ_API_KEY
# was ever set directly in PowerShell/Windows env vars (e.g. while testing
# another app), it can silently take priority over your .env file.
_pre_existing = os.environ.get("GROQ_API_KEY")
if _pre_existing:
    _masked_pre = (_pre_existing[:6] + "..." + _pre_existing[-4:]) if len(_pre_existing) > 10 else _pre_existing
    print(f"⚠️  Found GROQ_API_KEY already set in your shell/OS environment: {_masked_pre} "
          f"(length {len(_pre_existing)})")
    print("    config.py now forces .env to override this, but if the value above")
    print("    looks wrong, also remove it from your OS so it doesn't cause confusion:")
    print("    PowerShell (current session): Remove-Item Env:GROQ_API_KEY")
    print("    Permanent: Windows Settings > System > About > Advanced system settings")
    print("    > Environment Variables > delete GROQ_API_KEY there too.\n")

import config
from groq import Groq

print(f"Looking for .env at: {config.ENV_PATH}")
print(f"File exists: {config.ENV_PATH.exists()}")

# Scan the raw file for every GROQ_API_KEY line. If there are 2+, the LAST
# one wins when dotenv parses the file -- so a leftover placeholder line
# below your real key will silently override it.
if config.ENV_PATH.exists():
    raw_lines = config.ENV_PATH.read_text(encoding="utf-8-sig").splitlines()
    matches = [(i + 1, line) for i, line in enumerate(raw_lines) if line.strip().startswith("GROQ_API_KEY")]
    if len(matches) == 0:
        print("\n❌ No line starting with 'GROQ_API_KEY' found in .env at all.")
    elif len(matches) > 1:
        print(f"\n❌ Found {len(matches)} lines defining GROQ_API_KEY in .env -- only the LAST one is used:")
        for lineno, line in matches:
            value = line.split("=", 1)[1].strip() if "=" in line else ""
            masked = (value[:6] + "..." + value[-4:]) if len(value) > 10 else value
            print(f"   line {lineno}: GROQ_API_KEY={masked}")
        print("   Fix: delete all but one of these lines, keeping only your real key.")
    else:
        lineno, line = matches[0]
        value = line.split("=", 1)[1].strip() if "=" in line else ""
        masked = (value[:6] + "..." + value[-4:]) if len(value) > 10 else value
        print(f"\nFound exactly one GROQ_API_KEY line (line {lineno}): {masked}")

if not config.GROQ_API_KEY:
    print("\n❌ GROQ_API_KEY is empty after loading .env.")
    print("   - Check the file is named exactly '.env' (not '.env.example' or '.env.txt')")
    print("   - Check it sits in the same folder as this script")
    print("   - Check the line looks like: GROQ_API_KEY=gsk_xxxxxxxxxxxx (no quotes)")
    sys.exit(1)

masked = config.GROQ_API_KEY[:6] + "..." + config.GROQ_API_KEY[-4:]
print(f"\n✅ Key loaded: {masked} (length {len(config.GROQ_API_KEY)})")

print(f"Testing a real API call with model '{config.GROQ_MODEL}'...")

try:
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
        max_tokens=5,
    )
    print(f"\n✅ Success! Groq replied: {resp.choices[0].message.content!r}")
except Exception as e:
    print(f"\n❌ API call failed: {e}")
    print("\nIf this is a 401 Unauthorized, the key itself is being rejected.")
    print("Double check it's copied fully from https://console.groq.com/keys")
    print("and hasn't been revoked/regenerated since you copied it.")
    sys.exit(1)