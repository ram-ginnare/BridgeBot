"""
Standalone diagnostic: confirms the web search fallback (ddgs) actually
returns results, independent of the rest of the RAG pipeline.

Usage:
    python check_web_search.py
"""

from ddgs import DDGS

query = "Who is Narendra Modi"
print(f"Searching: {query!r}\n")

try:
    results = list(DDGS().text(query, max_results=5))
    if not results:
        print("❌ Got an empty result list. Possible causes:")
        print("   - Outbound network/firewall blocking requests to search backends")
        print("   - Temporary rate limiting -- wait a bit and retry")
        print("   - Corporate proxy/VPN interfering -- try DDGS(proxy='...') if you use one")
    else:
        print(f"✅ Got {len(results)} result(s):\n")
        for r in results:
            print(f"- {r.get('title')}\n  {r.get('href')}\n  {r.get('body', '')[:150]}...\n")
except Exception as e:
    print(f"❌ Search raised an exception: {e}")
    print("   If this mentions 'duckduckgo_search' or import errors, run:")
    print("   pip uninstall duckduckgo-search -y")
    print("   pip install -U ddgs")