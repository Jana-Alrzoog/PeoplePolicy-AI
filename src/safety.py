"""Security, governance, and agent boundary enforcement for PeoplePolicy AI."""

import time
import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import ALLOWED_TOOLS, MAX_TOOL_CALLS

logger = logging.getLogger("peoplepolicy.safety")


class SafeExecutionLog(BaseModel):
    """Sanitized technical audit log without sensitive data, API keys, or CoT."""
    tool_name: str
    status: str  # "SUCCESS", "REJECTED_UNAUTHORIZED", "VALIDATION_ERROR", "ERROR"
    duration_ms: float
    error_message: Optional[str] = None


class ToolCallCounter:
    """Tracks and limits the number of tool invocations within a single request turn."""

    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        self.max_calls = max_calls
        self.current_calls = 0

    def record_call(self) -> bool:
        if self.current_calls >= self.max_calls:
            return False
        self.current_calls += 1
        return True

    def reset(self) -> None:
        self.current_calls = 0


def is_tool_allowed(tool_name: str) -> bool:
    """Validate if requested tool name is in the strict security allowlist."""
    if not isinstance(tool_name, str):
        return False
    return tool_name.strip() in ALLOWED_TOOLS


def sanitize_log_message(message: str) -> str:
    """Remove potential secrets or accidental key leaks from logged messages."""
    redacted = str(message)
    # Mask common API key patterns if present
    import re
    redacted = re.sub(r"AIza[0-9A-Za-z\-_]+", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"bearer\s+[A-Za-z0-9\-_\.]+", "Bearer [REDACTED_TOKEN]", redacted, flags=re.IGNORECASE)
    return redacted


def create_safe_audit_entry(
    tool_name: str,
    status: str,
    start_time: float,
    error_message: Optional[str] = None,
) -> SafeExecutionLog:
    """Create a sanitized audit log entry showing execution duration and status only."""
    duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    sanitized_error = sanitize_log_message(error_message) if error_message else None
    return SafeExecutionLog(
        tool_name=tool_name,
        status=status,
        duration_ms=duration_ms,
        error_message=sanitized_error,
    )


# Common out-of-scope triggers (e.g. general geography, pop culture, non-HR programming)
OUT_OF_SCOPE_PATTERNS = [
    r"عاصمة\s+(فرنسا|مصر|أمريكا|بريطانيا|اليابان|ألمانيا|إيطاليا)",
    r"من\s+(فاز|ربح)\s+(بكأس|بالدوري|بالمباراة)",
    r"اكتب\s+لي\s+كود\s+(جافا|بايثون|سي)",
    r"ما\s+هو\s+الطقس",
    r"من\s+هو\s+رئيس",
]

def check_out_of_scope(query: str) -> Tuple[bool, Optional[str]]:
    """Detect if the user query is clearly outside the HR and workplace policies scope."""
    import re
    cleaned = query.strip()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            arabic_refusal = (
                "عذراً، أنا مساعد ذكي مخصص حصرياً لسياسات الموارد البشرية وإجراءات الإجازات "
                "في شركة أفق التجريبية. لا يمكنني الإجابة على استفسارات عامة خارج هذا النطاق."
            )
            return True, arabic_refusal
    return False, None


def sanitize_prompt_injection(content: str) -> str:
    """Defensive sanitation to ensure retrieved text is treated strictly as passive context."""
    # Strip any instructional override tags that might try to hijack LLM behavior
    markers = [
        "SYSTEM:", "INSTRUCTION:", "IGNORE PREVIOUS", "تجاهل التعليمات", 
        "أنت الآن", "YOU ARE NOW", "EXECUTE:", "ADMIN_OVERRIDE"
    ]
    sanitized = content
    for marker in markers:
        sanitized = sanitized.replace(marker, "[FILTERED_OVERRIDE]")
    return sanitized
