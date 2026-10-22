"""Unit tests for the 12 evaluation test cases via pytest."""

import pytest
from src.evaluation import EvaluationSuite


@pytest.fixture(scope="module")
def suite():
    return EvaluationSuite()


def test_eval_case_1_annual_leave_retrieval(suite):
    passed, actual, err = suite.run_case_1()
    assert passed is True, err


def test_eval_case_2_sick_leave_retrieval(suite):
    passed, actual, err = suite.run_case_2()
    assert passed is True, err


def test_eval_case_3_attendance_retrieval(suite):
    passed, actual, err = suite.run_case_3()
    assert passed is True, err


def test_eval_case_4_remote_work_retrieval(suite):
    passed, actual, err = suite.run_case_4()
    assert passed is True, err


def test_eval_case_5_leave_approval_retrieval(suite):
    passed, actual, err = suite.run_case_5()
    assert passed is True, err


def test_eval_case_6_leave_calc_standard(suite):
    passed, actual, err = suite.run_case_6()
    assert passed is True, err


def test_eval_case_7_leave_calc_exact_zero(suite):
    passed, actual, err = suite.run_case_7()
    assert passed is True, err


def test_eval_case_8_leave_calc_insufficient_balance(suite):
    passed, actual, err = suite.run_case_8()
    assert passed is True, err


def test_eval_case_9_reject_negative_requested_days(suite):
    passed, actual, err = suite.run_case_9()
    assert passed is True, err


def test_eval_case_10_reject_unexpected_schema_field(suite):
    passed, actual, err = suite.run_case_10()
    assert passed is True, err


def test_eval_case_11_reject_unauthorized_tool(suite):
    passed, actual, err = suite.run_case_11()
    assert passed is True, err


def test_eval_case_12_handle_out_of_scope_query(suite):
    passed, actual, err = suite.run_case_12()
    assert passed is True, err


# ==============================================================================
# EXTENDED EVALUATION CASES (Absence Deduction Cases 13 - 19)
# ==============================================================================

def test_eval_case_13_absence_deduction_retrieval(suite):
    passed, actual, err = suite.run_case_13()
    assert passed is True, err


def test_eval_case_14_absence_deduction_standard(suite):
    passed, actual, err = suite.run_case_14()
    assert passed is True, err


def test_eval_case_15_absence_deduction_zero_days(suite):
    passed, actual, err = suite.run_case_15()
    assert passed is True, err


def test_eval_case_16_absence_deduction_negative_inputs(suite):
    passed, actual, err = suite.run_case_16()
    assert passed is True, err


def test_eval_case_17_absence_deduction_excessive_days(suite):
    passed, actual, err = suite.run_case_17()
    assert passed is True, err


def test_eval_case_18_distinguish_paid_leave_from_unpaid(suite):
    passed, actual, err = suite.run_case_18()
    assert passed is True, err


def test_eval_case_19_block_unauthorized_payroll_action(suite):
    passed, actual, err = suite.run_case_19()
    assert passed is True, err


def test_eval_suite_original_12_completes(suite):
    """Verify original 12 evaluation test cases complete and pass 100%."""
    results = suite.run_original_12()
    assert len(results) == 12
    passed_count = sum(1 for r in results if r.passed)
    assert passed_count == 12


def test_eval_suite_run_all_completes(suite):
    """Verify full evaluation suite (original 12 + extended 7) reports 19 cases."""
    results = suite.run_all()
    assert len(results) == 19
    passed_count = sum(1 for r in results if r.passed)
    assert passed_count == 19
