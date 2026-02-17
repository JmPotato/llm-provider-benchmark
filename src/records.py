from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RequestRecord:
    run_id: str
    provider_name: str
    request_id: str
    prompt: str
    request_started_at: float
    first_token_at: float | None
    response_done_at: float
    output_tokens: int
    success: bool
    error_type: str | None
    error_message: str | None
    ttft_s: float | None
    tbt_s: float | None
    tpot_s: float | None
    e2e_s: float
    slo_passed: bool


@dataclass(slots=True)
class TokenEvent:
    run_id: str
    provider_name: str
    request_id: str
    token_index: int
    token_timestamp: float
    token_text: str
