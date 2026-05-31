from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Import the correct DDGS package (install with: pip install ddgs)
from ddgs import DDGS

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"⚠️ .env not found at {env_path}, checking default locations...")
    load_dotenv()  # fallback to default search

# Rest of your code follows...
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_engine")

# ---------------------------------------------------------------------------
# Pydantic Model (unchanged)
# ---------------------------------------------------------------------------
class CompetitionInfo(BaseModel):
    competition: str = Field(..., description="Exact or normalized name of the competition.")
    website: Optional[str] = Field(default=None, description="Official competition website URL.")
    registration_deadline: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    competition_start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    competition_end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    organizer: Optional[str] = Field(default=None, description="Company or group organizing the event.")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("website")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if not v or not isinstance(v, str):
            return None
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            v = "https://" + v
        parsed = urlparse(v)
        return v if parsed.netloc and "." in parsed.netloc else None

    @field_validator("registration_deadline", "competition_start_date", "competition_end_date", mode="before")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if not v or not isinstance(v, str):
            return None
        v = v.strip().split("T")[0].split()[0]
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

# ---------------------------------------------------------------------------
# Search Engine (no langchain-community)
# ---------------------------------------------------------------------------
class CompetitionSearchEngine:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_search_results: int = 5,
    ) -> None:
        self.logger = logging.getLogger("search_engine.CompetitionSearchEngine")
        self.max_search_results = max_search_results

        # --- Direct DuckDuckGo search (no langchain wrapper) ---
        try:
            self.ddgs = DDGS()
        except Exception as exc:
            raise RuntimeError("Failed to initialize DDGS (duckduckgo_search). Install with: pip install ddgs") from exc

        # --- Gemini LLM via LangChain ---
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("GOOGLE_API_KEY not provided. LLM extraction will fail.")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
        # Use structured output (native function calling)
        self.structured_llm = self.llm.with_structured_output(CompetitionInfo)

        # Prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Extract structured metadata about a case competition.
Return valid JSON with the following fields: competition, website, registration_deadline, competition_start_date, competition_end_date, organizer, confidence_score.
Use YYYY-MM-DD for dates. If a field cannot be found, use null. Confidence_score 0.0-1.0 based on snippet clarity."""),
            ("human", "Competition Name: {competition_name}\n\nWeb Search Results:\n{search_results}")
        ])

        # Chain: format results -> prompt -> structured LLM
        self.chain = (
            RunnablePassthrough.assign(
                search_results=lambda x: self._format_search_results(x["search_raw"])
            )
            | self.prompt
            | self.structured_llm
        )

        self.logger.info("Engine initialized with direct DuckDuckGo search.")

    # ---------------------------------------------------------------------
    # Search using DDGS directly
    # ---------------------------------------------------------------------
    def _execute_web_search(self, competition_name: str) -> List[Dict[str, str]]:
        """Run multiple queries and return raw results as list of dicts."""
        queries = [
            f'{competition_name} official website registration deadline',
            f'{competition_name} competition start end dates 2026',
            f'{competition_name} organizer company case competition'
        ]
        all_results = []
        for q in queries:
            try:
                # Use DDGS.text() to get search results
                results = list(self.ddgs.text(q, max_results=self.max_search_results))
                for r in results:
                    all_results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "link": r.get("href", ""),
                    })
            except Exception as e:
                self.logger.error("Search failed for '%s': %s", q, e)
        return all_results

    def _format_search_results(self, raw_results: List[Dict]) -> str:
        """Convert list of result dicts to a clean string for the prompt."""
        if not raw_results:
            return "No search results found."
        lines = []
        for r in raw_results:
            lines.append(f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}\nLink: {r.get('link', '')}\n")
        return "\n".join(lines)

    def _fallback_result(self, competition_name: str, reason: str) -> CompetitionInfo:
        self.logger.info("Fallback for '%s': %s", competition_name, reason)
        return CompetitionInfo(competition=competition_name, confidence_score=0.0)

    def _recalc_confidence(self, info: CompetitionInfo, raw_results: List[Dict]) -> float:
        """Optional: recalc confidence based on field presence and result richness."""
        field_weights = {"website": 0.25, "registration_deadline": 0.20,
                         "competition_start_date": 0.15, "competition_end_date": 0.15, "organizer": 0.25}
        presence = sum(weight for field, weight in field_weights.items() if getattr(info, field) is not None)
        text_len = sum(len(r.get("snippet", "")) for r in raw_results)
        richness = min(text_len / 4000, 1.0) * 0.15
        return round(min(1.0, presence * 0.7 + richness), 2)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def search(self, competition_name: str) -> Dict[str, Any]:
        self.logger.info("Searching for: %s", competition_name)
        try:
            raw_results = self._execute_web_search(competition_name)
            if not raw_results:
                return self._fallback_result(competition_name, "No search results").to_dict()

            final: CompetitionInfo = self.chain.invoke({
                "competition_name": competition_name,
                "search_raw": raw_results
            })
            # Overwrite confidence with recalculated score
            final.confidence_score = self._recalc_confidence(final, raw_results)
            return final.to_dict()
        except Exception as e:
            self.logger.exception("Pipeline error")
            return self._fallback_result(competition_name, str(e)).to_dict()

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = CompetitionSearchEngine()
    result = engine.search("Tata Crucible")
    print(json.dumps(result, indent=2, ensure_ascii=False))