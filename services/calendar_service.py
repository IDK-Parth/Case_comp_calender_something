# calendar_service.py

import os
import time
import random
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar"]
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds


def get_calendar_service():
    """Authenticate and return the Google Calendar service object."""
    creds = None

    # token.json stores the user's access and refresh tokens
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials, let the user log in
    if not creds or not creds.valid:
        # Ensure credentials.json is in the parent directory (case_compeletition_calender)
        creds_path = os.path.join(os.path.dirname(__file__), "..", "credentials.json")
        if not os.path.exists(creds_path):
            creds_path = "credentials.json"  # fallback to current dir
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def execute_with_retry(request, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """
    Execute a Google API request with exponential backoff retry logic.
    Handles HTTP 429 (rate limit) and 403 (quota/rate limit) errors.
    """
    backoff = initial_backoff
    for attempt in range(max_retries + 1):
        try:
            return request.execute()
        except HttpError as error:
            # Retry only on rate‑limit or quota errors
            if error.resp.status in (429, 403) and attempt < max_retries:
                # Add jitter to avoid synchronized retries
                sleep_time = backoff + random.uniform(0, 1)
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries+1}). "
                      f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                backoff *= 2  # exponential backoff
            else:
                # Re‑raise any other error or when retries exhausted
                raise


def create_event(service, title, date, description=""):
    """
    Create an all-day event on the primary calendar.
    :param service: authenticated Calendar service
    :param title: event summary
    :param date: string in YYYY-MM-DD format
    :param description: optional description / website URL
    """
    event = {
        "summary": title,
        "description": description,
        "start": {"date": date},
        "end": {"date": date},
    }
    request = service.events().insert(calendarId="primary", body=event)
    # Execute with built‑in retry logic for rate limits
    created_event = execute_with_retry(request)
    return created_event