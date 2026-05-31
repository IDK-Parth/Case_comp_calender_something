"""
Minimal competition search + domain classifier.
- Searches DuckDuckGo for competition details
- Uses Gemini (via LangChain) to extract metadata + categories (Agri, Tech, etc.)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS  # correct package: pip install duckduckgo-search

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# ----------------------------------------------------------------------
# Simple output models
# ----------------------------------------------------------------------
class CompetitionInfo(BaseModel):
    competition: str
    website: Optional[str] = None
    registration_deadline: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    organizer: Optional[str] = None
    categories: List[str] = Field(default_factory=list, description="e.g. ['Technology', 'Agriculture', 'Business', 'Social Impact']")
    confidence: float = Field(ge=0.0, le=1.0)

# ----------------------------------------------------------------------
# Search Engine (no over-engineering)
# ----------------------------------------------------------------------
class CompetitionSearchEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", max_results: int = 5):
        self.max_results = max_results
        self.ddgs = DDGS()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in .env")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
        )
    
    def _search(self, query: str) -> List[Dict]:
        """Raw DuckDuckGo search results."""
        try:
            return list(self.ddgs.text(query, max_results=self.max_results))
        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []
    
    def _extract_with_llm(self, competition_name: str, snippets: str) -> CompetitionInfo:
        """Single LLM call to get both metadata AND categories."""
        system_prompt = """You extract competition metadata from web snippets.
Return ONLY valid JSON with these fields:
- competition: exact name
- website: URL (null if none)
- registration_deadline: YYYY-MM-DD
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD
- organizer: name of organizing body
- categories: list of tags like ["Technology","Agriculture","Business","Science","Social Impact","Design","Data Science","Healthcare","Finance","Sustainability"]
- confidence: 0.0 to 1.0 (how certain you are)

Use null for missing fields. No extra text."""

        user_prompt = f"Competition: {competition_name}\n\nSearch snippets:\n{snippets}"
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse JSON from LLM response
        try:
            # Sometimes LLM wraps JSON in ```json ... ```
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            return CompetitionInfo(**data)
        except Exception as e:
            print(f"LLM parsing failed: {e}\nRaw response:\n{response.content}")
            return CompetitionInfo(competition=competition_name, confidence=0.0)
    
    def search(self, competition_name: str) -> Dict[str, Any]:
        print(f"🔎 Searching for: {competition_name}")
        
        # Combine multiple queries for richer snippets
        queries = [
            competition_name,
            f"{competition_name} competition dates deadline",
            f"{competition_name} organizer"
        ]
        all_snippets = []
        for q in queries:
            for res in self._search(q):
                snippet = res.get("body", "")
                if snippet:
                    all_snippets.append(f"- {snippet}")
        
        snippets_text = "\n".join(all_snippets[:15])  # keep token limit reasonable
        
        info = self._extract_with_llm(competition_name, snippets_text)
        return info.model_dump(exclude_none=True)

if __name__ == "__main__":
    engine = CompetitionSearchEngine()
    result = engine.search("Tata Crucible")
    print(json.dumps(result, indent=4, ensure_ascii=False))