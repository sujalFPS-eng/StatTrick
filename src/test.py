"""Smoke test for the Gemini key used by the AI Tactical Brief feature.

Fix vs. the original (#15 in the audit): never paste a live key into a source file that
gets committed. Reads it from the GEMINI_API_KEY environment variable instead.

Usage:
    GEMINI_API_KEY=... python3 src/test.py
"""
import os
import sys

from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("Set GEMINI_API_KEY in your environment first, e.g.:\n"
             "  export GEMINI_API_KEY=your_key_here\n"
             "  python3 src/test.py")

try:
    client = genai.Client(api_key=API_KEY)
    print("Connected. Models this key can use for generateContent:")
    print("-" * 50)
    for model in client.models.list():
        if "generateContent" in model.supported_actions:
            print(model.name)
except Exception as exc:
    sys.exit(f"Connection failed: {exc}")
