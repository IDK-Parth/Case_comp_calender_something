import os
import yaml
from datetime import date
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

# ---------- Configuration ----------
MODEL_NAME = "gemini-2.5-flash-lite"

# Load pricing from env (defaults: $0.10/1M input, $0.40/1M output)
INPUT_TOKEN_PRICE = float(os.getenv("GEMINI_INPUT_TOKEN_PRICE", "0.10e-6"))
OUTPUT_TOKEN_PRICE = float(os.getenv("GEMINI_OUTPUT_TOKEN_PRICE", "0.40e-6"))

# File for persistent token usage
USAGE_FILE = "config.yaml"

# ---------- Data Model ----------
class Competition(BaseModel):
    competition: str
    month: Optional[str] = None
    registration_deadline: Optional[str] = None
    website: Optional[str] = None
    event_type: Optional[str] = None

# ---------- Token Tracker (persistent, auto-reset daily) ----------
class TokenTracker:
    def __init__(self, usage_file: str = USAGE_FILE):
        self.usage_file = usage_file
        self.load_usage()

    def load_usage(self):
        """Load cumulative token usage from YAML file."""
        if os.path.exists(self.usage_file):
            with open(self.usage_file, "r") as f:
                data = yaml.safe_load(f) or {}
                usage = data.get("token_usage", {})
                self.input_tokens = usage.get("total_input_tokens", 0)
                self.output_tokens = usage.get("total_output_tokens", 0)
                self.requests = usage.get("total_requests", 0)
                self.last_reset = usage.get("last_reset_date")
        else:
            self.input_tokens = 0
            self.output_tokens = 0
            self.requests = 0
            self.last_reset = None

        # Optional: auto-reset daily if date changed
        today = date.today().isoformat()
        if self.last_reset != today:
            self.input_tokens = 0
            self.output_tokens = 0
            self.requests = 0
            self.last_reset = today
            self.save_usage()

    def save_usage(self):
        """Save current usage back to YAML."""
        data = {
            "token_usage": {
                "total_input_tokens": self.input_tokens,
                "total_output_tokens": self.output_tokens,
                "total_requests": self.requests,
                "last_reset_date": self.last_reset,
            }
        }
        with open(self.usage_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def add_usage(self, input_t: int, output_t: int):
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.requests += 1
        self.save_usage()   # persist after each addition

    def estimated_cost(self) -> float:
        """Compute cost using global token prices."""
        return (self.input_tokens * INPUT_TOKEN_PRICE) + (self.output_tokens * OUTPUT_TOKEN_PRICE)

    def summary(self):
        print(f"\n📊 Token Usage (actual from API):")
        print(f"   Requests: {self.requests}")
        print(f"   Input tokens: {self.input_tokens:,}")
        print(f"   Output tokens: {self.output_tokens:,}")
        print(f"   Estimated cost: ${self.estimated_cost():.6f}")

# ---------- LLM Setup ----------
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=2,
)
structured_llm = llm.with_structured_output(Competition, method="json_mode")

# ---------- Main Extraction ----------
def extract_competitions(text: str, tracker: TokenTracker) -> List[Competition]:
    prompt = f"Extract all competitions from the text below. Return a list of competitions.\n\nText:\n{text}"
    response = structured_llm.invoke(prompt)

    # Get actual token usage from LangChain response
    usage = response.usage_metadata  # dict with 'input_tokens', 'output_tokens'
    tracker.add_usage(usage["input_tokens"], usage["output_tokens"])

    # Ensure we always return a list
    if not isinstance(response, list):
        response = [response]
    return response

# ---------- Run ----------
if __name__ == "__main__":
    sample_text = """
    Flipkart GRID 7.0
    Registration closes August 15
    Website: flipkart.com/grid

    HUL LIME
    September
    Website: hul.com/lime

    Google Code Jam
    May 15 - June 30
    Website: codingcompetitions.withgoogle.com
    """

    tracker = TokenTracker()
    competitions = extract_competitions(sample_text, tracker)

    print("\n🏆 Extracted Competitions:")
    for comp in competitions:
        print(f"  - {comp.competition} (deadline: {comp.registration_deadline})")

    tracker.summary()