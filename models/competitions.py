from pydantic import BaseModel
from typing import Optional

class Competition(BaseModel):
    competition: str
    month: Optional[str]
    registration_deadline: Optional[str]
    website: Optional[str]
    event_type: Optional[str]