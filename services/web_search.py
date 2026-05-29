from duckduckgo_search import DDGS

def search_registration(comp_name):

    query = f"{comp_name} registration deadline official website"

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    return results