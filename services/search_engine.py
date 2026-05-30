from dotenv import load_dotenv
import os
import time
import json
from queue import Queue
from threading import Lock
from datetime import datetime
from typing import List, Optional, Dict, Any
#from langchain.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from google.genai import Client

load_dotenv()

# Configuration
MODEL_NAME = "gemini-2.5-flash-lite"  # Fastest & cheapest model: $0.10/1M input, $0.40/1M output
INPUT_TOKEN_PRICE = 0.10 / 1_000_000  # $0.10 per million input tokens
OUTPUT_TOKEN_PRICE = 0.40 / 1_000_000  # $0.40 per million output tokens

# Token tracking class
class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.lock = Lock()
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        with self.lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1
    
    def get_estimated_cost(self) -> float:
        input_cost = self.total_input_tokens * INPUT_TOKEN_PRICE
        output_cost = self.total_output_tokens * OUTPUT_TOKEN_PRICE
        return input_cost + output_cost
    
    def display_status(self):
        print(f"\n📊 Token Usage Summary:")
        print(f"   Total Requests: {self.total_requests}")
        print(f"   Total Input Tokens: {self.total_input_tokens:,}")
        print(f"   Total Output Tokens: {self.total_output_tokens:,}")
        print(f"   Estimated Cost: ${self.get_estimated_cost():.6f}")
        print(f"   Total Cost (combined): ${(self.get_estimated_cost()):.6f}")

# FIFO Request Queue with Rate Limiting
class RateLimitedQueue:
    def __init__(self, requests_per_minute: int = 15):
        self.queue = Queue()
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.lock = Lock()
        self.token_tracker = TokenTracker()
        
    def add_request(self, prompt: str, callback):
        """Add a request to the FIFO queue"""
        self.queue.put((prompt, callback))
        
    def process_queue(self):
        """Process requests with rate limiting and FIFO ordering"""
        while not self.queue.empty():
            prompt, callback = self.queue.get()
            self._wait_if_needed()
            self._process_request(prompt, callback)
    
    def _wait_if_needed(self):
        """Implement rate limiting - 15 requests per minute"""
        with self.lock:
            current_time = time.time()
            # Remove timestamps older than 60 seconds
            self.request_timestamps = [ts for ts in self.request_timestamps 
                                       if current_time - ts < 60]
            
            if len(self.request_timestamps) >= self.requests_per_minute:
                oldest = min(self.request_timestamps)
                wait_time = 60 - (current_time - oldest)
                if wait_time > 0:
                    print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time + 0.1)
            
            self.request_timestamps.append(time.time())
    
    def _process_request(self, prompt: str, callback):
        """Process a single request and update token tracking"""
        try:
            # Pre-call token estimation using countTokens API
            input_tokens = self.estimate_tokens(prompt)
            print(f"📝 Estimated input tokens: {input_tokens:,}")
            
            # Make the API call
            result = callback(prompt)
            
            # Estimate output tokens (simplified - use actual response tokens)
            output_tokens = len(str(result).split()) // 0.75  # Rough estimate
            print(f"📤 Estimated output tokens: {output_tokens:,}")
            
            # Update token tracker
            self.token_tracker.add_usage(input_tokens, output_tokens)
            self.token_tracker.display_status()
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            return None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using simple math (approx 0.75 words per token)"""
        word_count = len(text.split())
        return int(word_count // 0.75) or 100  # Minimum 100 tokens for safety
    
    def get_remaining_quota(self, daily_limit: int = 1500) -> int:
        """Calculate remaining daily quota based on current usage"""
        remaining = daily_limit - self.token_tracker.total_requests
        return max(0, remaining)

# Define the data model
class Competition(BaseModel):
    competition: str = Field(description="Name of the competition")
    month: Optional[str] = Field(description="Month when competition occurs", default=None)
    registration_deadline: Optional[str] = Field(description="Registration deadline", default=None)
    website: Optional[str] = Field(description="Official website URL", default=None)
    event_type: Optional[str] = Field(description="Type of event", default=None)

# Initialize the model (fastest & cheapest)
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=2,
)

# Create structured output handler
structured_llm = llm.with_structured_output(Competition, method="json_mode")

# Initialize rate-limited queue
rate_limiter = RateLimitedQueue(requests_per_minute=15)

def process_competition_extraction(prompt: str):
    """Callback function for processing competition extraction"""
    response = structured_llm.invoke(prompt)
    if not isinstance(response, list):
        response = [response]
    return response

# Sample text with multiple competitions
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

prompt = f"""
Extract all competitions from the text below. Return a list of competitions.

Text:
{sample_text}
"""

# Add request to FIFO queue
rate_limiter.add_request(prompt, process_competition_extraction)

# Process all queued requests
print("🚀 Starting to process competition extraction requests...")
print(f"📊 Using model: {MODEL_NAME}")
print(f"⏱️  Rate limit: 15 requests per minute")

# Process the queue
rate_limiter.process_queue()

# Display final quota status
remaining = rate_limiter.get_remaining_quota(1500)
print(f"\n🎯 Remaining daily quota: {remaining} requests")