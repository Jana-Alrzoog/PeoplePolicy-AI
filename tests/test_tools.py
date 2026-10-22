"""Unit tests for validated HR leave calculation tools."""

import pytest
import math
from pydantic import ValidationError
from src.tools import (
    calculate_leave_balance,
    execute_tool,
    LeaveCalculationInput,
)


def test_calculate_leave_balance_standard():
    """Verify standard valid calculation: 15 - 6 = 9."""
    result = calculate_leave_balance(available_balance=15.0, requested_days=6.0)
    assert result.available_balance == 15.0
    assert result.requested_days == 6.0
    assert result.remaining_balance == 9.0
    assert result.eligible is True
    assert "مؤهل" in result.decision_reason


def test_calculate_leave_balance_exact_match():
    """Verify exact balance calculation: 5 - 5 = 0."""
    result = calculate_leave_balance(available_balance=5.0, requested_days=5.0)
    assert result.remaining_balance == 0.0
    assert result.eligible is True


def test_calculate_leave_balance_insufficient():
    """Verify insufficient balance: 4 available, 10 requested."""
    result = calculate_leave_balance(available_balance=4.0, requested_days=10.0)
    assert result.eligible is False
    assert result.remaining_balance >= 0.0  # Does not produce negative approved balance
    assert "غير مؤهل" in result.decision_reason


def test_decimal_precision_avoidance():
    """Verify that Decimal arithmetic avoids float precision quirks (e.g. 10.3 - 0.1)."""
    result = calculate_leave_balance(available_balance=10.3, requested_days=0.1)
    assert result.remaining_balance == 10.2
    assert result.eligible is True


def test_validation_rejects_negative_requested_days():
    """Verify that negative requested days are strictly rejected."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=10.0, requested_days=-1.0)


def test_validation_rejects_negative_available_balance():
    """Verify that negative available balance is strictly rejected."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=-5.0, requested_days=2.0)


def test_validation_rejects_zero_requested_days():
    """Verify that 0 requested days is rejected (must be > 0)."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=10.0, requested_days=0.0)


def test_validation_rejects_booleans():
    """Verify booleans are strictly rejected even though isinstance(True, int) is True in Python."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=True, requested_days=2.0)

    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=10.0, requested_days=False)


def test_validation_rejects_strings():
    """Verify strings masquerading as numbers are rejected in strict mode."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance="15", requested_days=2.0)


def test_validation_rejects_nan_and_inf():
    """Verify NaN and Infinity are strictly rejected."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=float("nan"), requested_days=2.0)

    with pytest.raises(ValidationError):
        LeaveCalculationInput(available_balance=10.0, requested_days=float("inf"))


def test_validation_rejects_extra_fields():
    """Verify unexpected extra fields are forbidden by schema."""
    with pytest.raises(ValidationError):
        LeaveCalculationInput(
            available_balance=10.0,
            requested_days=2.0,
            malicious_field="injection",
        )


def test_execute_tool_success_dispatch():
    """Verify execute_tool wrapper produces structured success output."""
    res, log = execute_tool(
        "calculate_leave_balance",
        {"available_balance": 20, "requested_days": 5},
    )
    assert res["success"] is True
    assert res["data"]["remaining_balance"] == 15.0
    assert log.status == "SUCCESS"
    assert log.duration_ms >= 0


def test_execute_tool_validation_error_dispatch():
    """Verify execute_tool captures validation errors cleanly."""
    res, log = execute_tool(
        "calculate_leave_balance",
        {"available_balance": -10, "requested_days": 5},
    )
    assert res["success"] is False
    assert res["error_type"] == "VALIDATION_ERROR"
    assert log.status == "VALIDATION_ERROR"
