"""Trusted structured inputs keep remote voice distinct from keyboard text."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class WebRequest:
    """Server-authenticated input identity and active conversation revision."""

    request_id: str
    device_id: str
    conversation_handle: str
    revision: int
    input_kind: Literal["text", "voice"]
    text: str = ""
    attachment_ids: tuple[str, ...] = ()
    owner_key: str = field(default="owner", repr=False)
