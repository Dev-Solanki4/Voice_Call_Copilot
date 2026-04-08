from base64 import b64decode
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Optional, TypedDict
import requests
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from .exceptions import MeetingInitializationError
from enum import Enum

logger = logging.getLogger(__name__)
KEY = b"YSJ!gMW!Y1Eu8j1NTb^rZyQiNLplYz*n"[
    :32
]  # AES key must be 16, 24, or 32 bytes long
IV = b"y^LKkEGk0FwxjQ#B"[:16]  # IV must be 16 bytes long


class ReliabilityModes(Enum):
    RELIABLE = "RELIABLE"
    UNRELIABLE = "UNRELIABLE"

@dataclass
class ServerConfig(TypedDict):
    base_url: str = None
    ice_servers: list[Any] = field(default_factory=list)
    observability_jwt: str = None
    traces: dict[str, Any] = field(default_factory=dict)
    meta_data: dict[str, Any] = field(default_factory=dict)
    signaling_url = str = None


@dataclass
class SummaryConfig(TypedDict, total=False):
    enabled: Optional[bool] = False
    prompt: Optional[str] = None


@dataclass
class TranscriptionConfig(TypedDict, total=False):
    webhook_url: Optional[bool] = False
    summary: Optional[SummaryConfig] = None
    language: Optional[str] = None


@dataclass
class PostTranscriptionConfig(TypedDict, total=False):
    enabled: Optional[bool] = False
    summary: Optional[SummaryConfig] = None


@dataclass
class RecordingConfig(TypedDict, total=False):
    dir_path: Optional[str]
    transcription: Optional[PostTranscriptionConfig]
    config: Optional[Dict]
    webhook_url: Optional[str]


@dataclass
class PubSubPublishConfig(TypedDict, total=False):
    topic: str
    message: str
    options: Optional[Dict]
    payload: Any


@dataclass
class PubSubSubscribeConfig(TypedDict, total=False):
    topic: str
    cb: Callable


def get_server_config(room_id, secret, signaling_base_url) -> ServerConfig:
    host = "api.videosdk.live"

    ice_servers = []
    observability_jwt = None
    traces = {}
    logs = {}
    meta_data = {}
    signaling_url = None

    try:
        logger.debug(
            f"getting server config for room :: {room_id} and url :: {signaling_base_url}"
        )
        response = requests.post(
            f"https://{signaling_base_url}/infra/v1/meetings/server-init-config",
            headers={
                "Authorization": secret,
                "Content-Type": "application/json",
            },
            json={"roomId": room_id},
        )
        response.raise_for_status()
        data = response.json().get("data", {})
    except requests.RequestException as e:
        logger.debug("error while api request for init-config")
        raise MeetingInitializationError(e)

    logger.debug(f"server config found for room :: {room_id}")
    if data:
        host = data.get("baseUrl", host)
        observability = data.get("observability", {})
        ice_servers = aes_decrypt(data.get("iceServers", ""))
        observability_jwt = observability.get("jwt")
        traces = observability.get("traces", {})
        logs = observability.get("logs", {})
        meta_data = observability.get("metaData", {})
        signaling_url = data.get("signalingUrl")

    return {
        "base_url": host,
        "ice_servers": ice_servers,
        "observability_Jwt": observability_jwt,
        "traces": traces,
        "logs": logs,
        "metaData": meta_data,
        "signaling_url": signaling_url,
    }


def aes_decrypt(value):
    try:
        # Base64 decode the input value
        encrypted_bytes = b64decode(value)
        cipher = AES.new(KEY, AES.MODE_CBC, IV)
        decrypted_bytes = unpad(
            cipher.decrypt(encrypted_bytes), AES.block_size, style="pkcs7"
        )
        plaintext = decrypted_bytes.decode("utf-8")
        return json.loads(plaintext)
    except (ValueError, KeyError) as e:
        logger.error("Error during decryption", exc_info=e)
        raise MeetingInitializationError(e)
