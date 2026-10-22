"""Validated HR calculation tools and safe tool execution registry for PeoplePolicy AI."""

import time
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.safety import is_tool_allowed, create_safe_audit_entry, SafeExecutionLog


class LeaveCalculationInput(BaseModel):
    """Strict input schema for annual leave balance calculation."""
    model_config = ConfigDict(
        extra="forbid",        # Reject unexpected fields
        strict=True,           # Reject implicit type coercions (like string to float or bool to int)
    )

    available_balance: float = Field(
        ...,
        description="رصيد الإجازة السنوية المتاح حالياً للموظف (يجب أن يكون 0 أو أكبر)",
    )
    requested_days: float = Field(
        ...,
        description="عدد أيام الإجازة السنوية المراد التمتع بها (يجب أن يكون أكبر من 0)",
    )

    @field_validator("available_balance", "requested_days", mode="before")
    @classmethod
    def validate_finite_and_type(cls, value: Any, info) -> Any:
        field_name = info.field_name
        # Reject booleans explicitly (in Python, isinstance(True, int/float) can be True)
        if isinstance(value, bool):
            field_label = "الرصيد المتاح" if field_name == "available_balance" else "الأيام المطلوبة"
            raise ValueError(f"قيمة حقل '{field_label}' غير صالحة: القيم المنطقية (Boolean) غير مقبولة.")

        # Reject strings masquerading as numbers in strict validation
        if isinstance(value, str):
            field_label = "الرصيد المتاح" if field_name == "available_balance" else "الأيام المطلوبة"
            raise ValueError(f"قيمة حقل '{field_label}' غير صالحة: يجب إدخال رقم صريح وليس نصاً.")

        # Validate numeric type
        if not isinstance(value, (int, float, Decimal)):
            field_label = "الرصيد المتاح" if field_name == "available_balance" else "الأيام المطلوبة"
            raise ValueError(f"قيمة حقل '{field_label}' يجب أن تكون قيمة رقمية صحيحة.")

        # Reject NaN and Infinity
        if isinstance(value, (float, Decimal)):
            if math.isnan(value) or math.isinf(value):
                field_label = "الرصيد المتاح" if field_name == "available_balance" else "الأيام المطلوبة"
                raise ValueError(f"قيمة حقل '{field_label}' غير مقبولة (قيمة لا نهائية أو غير معرّفة NaN).")

        return float(value)

    @field_validator("available_balance")
    @classmethod
    def validate_available_balance(cls, value: float) -> float:
        if value < 0:
            raise ValueError("الرصيد المتاح لا يمكن أن يكون سالباً. القيمة الدنيا هي 0.")
        return value

    @field_validator("requested_days")
    @classmethod
    def validate_requested_days(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("عدد الأيام المطلوبة يجب أن يكون رقماً موجباً أكبر تماماً من الصفر (0).")
        return value


class LeaveCalculationOutput(BaseModel):
    """Standardized response schema for leave balance calculation."""
    available_balance: float
    requested_days: float
    remaining_balance: float
    eligible: bool
    decision_reason: str
    is_administrative_approval: bool = False
    disclaimer: str = "هذا الحساب يمثل أهلية الرصيد الحسابية فقط، ولا يُعد موافقة إدارية نهائية على الإجازة."


class AbsenceDeductionInput(BaseModel):
    """Strict input schema for unpaid absence deduction calculation."""
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    base_salary: float = Field(
        ...,
        description="الراتب الأساسي الشهري بالريال السعودي (يجب أن يكون أكبر من 0)",
    )
    unpaid_absence_days: float = Field(
        ...,
        description="عدد أيام الغياب غير المدفوعة (يجب أن يكون 0 أو أكبر ولا يتجاوز 30 يوماً)",
    )

    @field_validator("base_salary", "unpaid_absence_days", mode="before")
    @classmethod
    def validate_finite_and_type(cls, value: Any, info) -> Any:
        field_name = info.field_name
        field_label = "الراتب الأساسي" if field_name == "base_salary" else "أيام الغياب غير المدفوعة"

        if isinstance(value, bool):
            raise ValueError(f"قيمة حقل '{field_label}' غير صالحة: القيم المنطقية (Boolean) غير مقبولة.")

        if isinstance(value, str):
            raise ValueError(f"قيمة حقل '{field_label}' غير صالحة: يجب إدخال رقم صريح وليس نصاً.")

        if not isinstance(value, (int, float, Decimal)):
            raise ValueError(f"قيمة حقل '{field_label}' يجب أن تكون قيمة رقمية صحيحة.")

        if isinstance(value, (float, Decimal)):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"قيمة حقل '{field_label}' غير مقبولة (قيمة لا نهائية أو غير معرّفة NaN).")

        return float(value)

    @field_validator("base_salary")
    @classmethod
    def validate_base_salary(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("الراتب الأساسي يجب أن يكون رقماً موجباً أكبر تماماً من الصفر (0).")
        return value

    @field_validator("unpaid_absence_days")
    @classmethod
    def validate_absence_days(cls, value: float) -> float:
        if value < 0:
            raise ValueError("عدد أيام الغياب لا يمكن أن يكون سالباً. القيمة الدنيا هي 0.")
        if value > 30:
            raise ValueError(
                "عدد أيام الغياب غير المدفوعة لا يمكن أن يتجاوز 30 يوماً في الشهر التقويمي الواحد؛ "
                "ولا يُسمح بأن يتجاوز مبلغ الخصم إجمالي الراتب الأساسي."
            )
        return value


class AbsenceDeductionOutput(BaseModel):
    """Standardized response schema for absence deduction calculation."""
    base_salary: float
    unpaid_absence_days: float
    daily_wage: float
    estimated_deduction: float
    estimated_remaining_salary: float
    is_deduction_applied: bool
    explanation: str
    currency: str = "SAR"
    disclaimer: str = (
        "هذا الحساب تقديري واسترشادي مبني على المعادلة الافتراضية لشركة أفق التجريبية، "
        "ولا يمثل مسير رواتب رسمي أو قراراً تأديبياً ملزماً قانونياً."
    )


def calculate_absence_deduction(
    base_salary: float,
    unpaid_absence_days: float,
) -> AbsenceDeductionOutput:
    """
    Calculate estimated absence deduction using Decimal arithmetic.
    Formula:
      daily_wage = base_salary / 30
      estimated_deduction = daily_wage * unpaid_absence_days
      estimated_remaining_salary = base_salary - estimated_deduction
    """
    try:
        dec_salary = Decimal(str(base_salary))
        dec_days = Decimal(str(unpaid_absence_days))
        dec_divisor = Decimal("30")
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValueError(f"خطأ في الدقة الحسابية: {e}")

    # Calculate daily wage (base_salary / 30)
    dec_daily_wage = (dec_salary / dec_divisor).quantize(Decimal("0.01"))
    dec_deduction = (dec_daily_wage * dec_days).quantize(Decimal("0.01"))

    # Cap deduction at base salary
    if dec_deduction > dec_salary:
        dec_deduction = dec_salary

    dec_remaining = dec_salary - dec_deduction

    float_salary = float(dec_salary)
    float_days = float(dec_days)
    float_daily = float(dec_daily_wage)
    float_deduction = float(dec_deduction)
    float_remaining = float(dec_remaining)

    if float_days == 0:
        is_applied = False
        explanation = (
            f"لا يوجد أيام غياب غير مدفوعة (0 يوم). أجر اليوم التقديري هو ({float_daily}) ريال سعودي، "
            f"ولم يتم تطبيق أي خصم. الراتب المتبقي التقديري هو ({float_remaining}) ريال سعودي."
        )
    else:
        is_applied = True
        explanation = (
            f"بناءً على راتب أساسي قدره ({float_salary}) ر.س وعدد ({float_days}) يوم غياب غير مدفوع: "
            f"أجر اليوم الواحد = {float_salary} ÷ 30 = {float_daily} ر.س. "
            f"الخصم التقديري = {float_daily} × {float_days} = {float_deduction} ر.س. "
            f"الراتب المتبقي المتوقع = {float_remaining} ر.س."
        )

    return AbsenceDeductionOutput(
        base_salary=float_salary,
        unpaid_absence_days=float_days,
        daily_wage=float_daily,
        estimated_deduction=float_deduction,
        estimated_remaining_salary=float_remaining,
        is_deduction_applied=is_applied,
        explanation=explanation,
    )


def calculate_leave_balance(
    available_balance: float,
    requested_days: float,
) -> LeaveCalculationOutput:
    """Execute high-precision Decimal calculation for leave balance eligibility."""
    # Convert to Decimal for exact precision without binary floating point discrepancies
    try:
        dec_available = Decimal(str(available_balance))
        dec_requested = Decimal(str(requested_days))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValueError(f"خطأ في الدقة الحسابية: {e}")

    if dec_requested <= dec_available:
        dec_remaining = dec_available - dec_requested
        eligible = True
        remaining_float = float(dec_remaining)
        reason = (
            f"الطلب مؤهل حسابياً: الرصيد المتاح ({float(dec_available)}) يغطي "
            f"الأيام المطلوبة ({float(dec_requested)}). الرصيد المتبقي المتوقع هو ({remaining_float}) يوماً."
        )
    else:
        # Rejection: insufficient balance, do not produce a negative remaining balance as an approved result
        eligible = False
        remaining_float = float(dec_available)
        shortage = float(dec_requested - dec_available)
        reason = (
            f"الطلب غير مؤهل لعدم كفاية الرصيد: الرصيد المتاح ({float(dec_available)}) أقل من "
            f"الأيام المطلوبة ({float(dec_requested)}). هناك نقص قدره ({shortage}) يوماً، "
            "ولايسمح النظام برصيد سالب."
        )

    return LeaveCalculationOutput(
        available_balance=float(dec_available),
        requested_days=float(dec_requested),
        remaining_balance=remaining_float,
        eligible=eligible,
        decision_reason=reason,
    )


def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Tuple[Dict[str, Any], SafeExecutionLog]:
    """Secure dispatcher executing allowed tools with strict validation and audit logging."""
    start_time = time.perf_counter()

    # Step 1: Security allowlist check
    if not is_tool_allowed(tool_name):
        log_entry = create_safe_audit_entry(
            tool_name=tool_name,
            status="REJECTED_UNAUTHORIZED",
            start_time=start_time,
            error_message=f"أداة غير مصرح بها: {tool_name}",
        )
        return {
            "success": False,
            "error_type": "UNAUTHORIZED_TOOL",
            "message": f"عملية مرفوضة أمنياً: الأداة '{tool_name}' غير مصرح بها في نظام الموارد البشرية.",
        }, log_entry

    # Step 2: Input validation and execution
    if tool_name == "calculate_leave_balance":
        try:
            validated_input = LeaveCalculationInput(**arguments)
            result = calculate_leave_balance(
                available_balance=validated_input.available_balance,
                requested_days=validated_input.requested_days,
            )
            log_entry = create_safe_audit_entry(
                tool_name=tool_name,
                status="SUCCESS",
                start_time=start_time,
            )
            return {
                "success": True,
                "data": result.model_dump(),
            }, log_entry
        except Exception as e:
            error_str = str(e)
            log_entry = create_safe_audit_entry(
                tool_name=tool_name,
                status="VALIDATION_ERROR",
                start_time=start_time,
                error_message=error_str,
            )
            return {
                "success": False,
                "error_type": "VALIDATION_ERROR",
                "message": f"فشل التحقق من مدخلات أداة الحساب: {error_str}",
            }, log_entry

    if tool_name == "calculate_absence_deduction":
        try:
            validated_absence = AbsenceDeductionInput(**arguments)
            result = calculate_absence_deduction(
                base_salary=validated_absence.base_salary,
                unpaid_absence_days=validated_absence.unpaid_absence_days,
            )
            log_entry = create_safe_audit_entry(
                tool_name=tool_name,
                status="SUCCESS",
                start_time=start_time,
            )
            return {
                "success": True,
                "data": result.model_dump(),
            }, log_entry
        except Exception as e:
            error_str = str(e)
            log_entry = create_safe_audit_entry(
                tool_name=tool_name,
                status="VALIDATION_ERROR",
                start_time=start_time,
                error_message=error_str,
            )
            return {
                "success": False,
                "error_type": "VALIDATION_ERROR",
                "message": f"فشل التحقق من مدخلات أداة خصم الغياب: {error_str}",
            }, log_entry

    # Fallback for unrecognized allowed tool
    log_entry = create_safe_audit_entry(
        tool_name=tool_name,
        status="ERROR",
        start_time=start_time,
        error_message="Not implemented",
    )
    return {
        "success": False,
        "error_type": "NOT_IMPLEMENTED",
        "message": f"الأداة '{tool_name}' غير مكتملة التنفيذ.",
    }, log_entry
