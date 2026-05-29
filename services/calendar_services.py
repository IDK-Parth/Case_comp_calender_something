from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def create_event(service, comp):

    event = {
        'summary': comp['competition'],
        'description': comp.get('website', ''),
        'start': {
            'date': comp['registration_deadline']
        },
        'end': {
            'date': comp['registration_deadline']
        },
    }

    service.events().insert(
        calendarId='primary',
        body=event
    ).execute()