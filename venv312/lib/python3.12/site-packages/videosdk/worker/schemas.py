from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class NotificationTypes:
    END_INTERACTION = "end_interaction"
    MATCH_INTENT = "match_intent"
    SET_GUIDELINES = "set_guidelines"
    GET_VISION = "get_vision"
    BROADCAST = "broadcast"


class CharacterResponseStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class StateRequest(BaseModel):
    interaction_id: str
    current_state_name: Optional[str] = None
    current_state_description: Optional[str] = None
    current_state_args: Optional[Dict[str, Any]] = {}
    grpc_service_url: Optional[str] = None


class StateResponse(BaseModel):
    interaction_id: str
    next_state: str
    next_state_description: str
    next_state_params: Dict[str, Any]
    response_status: CharacterResponseStatus
    response_message: str


class MatchIntentPayload(BaseModel):
    event: str = Field(default=NotificationTypes.MATCH_INTENT)
    interaction_id: str
    args: Optional[Dict[str, Any]] = None


class MatchIntentResult(BaseModel):
    success: bool
    score: float


class SetGuidelinesPayload(BaseModel):
    event: str = Field(default=NotificationTypes.SET_GUIDELINES)
    interaction_id: str
    args: Optional[Dict[str, Any]] = None


class SetGuidelinesResult(BaseModel):
    success: bool
    message: str


class EndInteractionPayload(BaseModel):
    event: str = Field(default=NotificationTypes.END_INTERACTION)
    interaction_id: str
    args: Optional[Dict[str, Any]] = None


class VisionRequestPayload(BaseModel):
    event: str = Field(default=NotificationTypes.GET_VISION)
    interaction_id: str
    args: Optional[Dict[str, Any]] = None


class CharacterVisionResult(BaseModel):
    success: bool
    message: str
    image: Optional[bytes] = None


class BroadcastPayload(BaseModel):
    event: str = Field(default=NotificationTypes.BROADCAST)
    interaction_id: str
    args: Optional[Dict[str, Any]] = None
