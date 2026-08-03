"""web_fallback

Provides a small wrapper around the ``ddgs`` (DuckDuckGo) search client
that the RAG pipeline can call at runtime. Exposes two functions used by
the app:

- ``web_search(query)`` -> list[dict]
- ``format_web_context(web_results)`` -> str

The original diagnostic behaviour is preserved under ``__main__`` so you
can run this file directly for quick manual checks.
"""

from typing import List, Dict, Optional
import logging
import config

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: Optional[int] = None, proxy: Optional[str] = None) -> List[Dict]:
    """Perform a web search and return a list of result dicts.

    Each result dict contains at least: ``title``, ``href`` and ``body``.
    On error this returns an empty list and logs the exception.
    """
    if max_results is None:
        max_results = config.WEB_SEARCH_MAX_RESULTS

    try:
        # Import lazily so the module can be imported even if ddgs isn't
        # available in the current environment (useful for CI/tests).
        from ddgs import DDGS
        ddgs = DDGS(proxy=proxy) if proxy else DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        # Normalize results to a predictable shape
        out = []
        for r in results:
            out.append({
                "title": r.get("title") or "",
                "href": r.get("href") or "",
                "body": r.get("body") or "",
            })
        return out
    except Exception as e:
        logger.exception("web_search failed")
        return []


def format_web_context(web_results: List[Dict]) -> str:
    """Format a list of web search results into a single string used as
    context for the generation prompt.
    """
    if not web_results:
        return ""

    blocks = []
    for r in web_results:
        title = r.get("title") or "(no title)"
        href = r.get("href") or ""
        body = r.get("body") or ""
        # Keep bodies reasonably short to avoid overly long prompts
        snippet = (body[:1000] + "...") if len(body) > 1000 else body
        blocks.append(f"[Source: {title}]\n{snippet}\nURL: {href}")

    return "\n\n".join(blocks)


if __name__ == "__main__":
    # Simple diagnostic, preserved from the original script.
    import sys

    query = "Who is Narendra Modi"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print(f"Searching: {query!r}\n")
    try:
        results = web_search(query)
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