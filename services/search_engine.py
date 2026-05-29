from duckduckgo_search import DDGS

def search_competition(query, max_results=5):
    """Return a list of dicts with keys: title, link, snippet."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title"),
                "link": r.get("href"),
                "snippet": r.get("body")
            })
    return results

def search_registration(comp_name):
    """Convenience wrapper for deadline/organizer search."""
    query = f"{comp_name} registration deadline official website"
    return search_competition(query, max_results=5)