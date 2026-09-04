from __future__ import annotations

from pathlib import Path

import pytest

from tools.validate_test_report import ReportGateError, main, parse_junit, validate_report


def _write_report(tmp_path: Path, content: str) -> Path:
    report = tmp_path / "report.xml"
    report.write_text(content, encoding="utf-8")
    return report


def test_parse_junit_counts_executed_and_skipped_cases(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuites tests="4" failures="1" errors="0" skipped="1">
          <testsuite name="pytest" tests="4">
            <testcase classname="suite" name="pass" />
            <testcase classname="suite" name="fail"><failure message="broken" /></testcase>
            <testcase classname="suite" name="skip">
              <skipped message="optional backend disabled" />
            </testcase>
            <testcase classname="suite" name="notrun" status="notrun" />
          </testsuite>
        </testsuites>
        """,
    )

    summary = parse_junit(report)

    assert summary.total == 4
    assert summary.executed == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.errors == 0
    assert summary.skipped == 2
    assert summary.required_skipped == 2
    assert summary.skip_reasons == ("optional backend disabled", "unspecified skip")


def test_validate_report_allows_only_explicit_skip_reasons(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="2" failures="0" errors="0" skipped="1">
          <testcase classname="suite" name="pass" />
          <testcase classname="suite" name="skip">
            <skipped message="optional resource is absent" />
          </testcase>
        </testsuite>
        """,
    )

    summary = validate_report(
        report,
        expected_total=2,
        min_executed=1,
        allow_skip_regex=(r"optional resource",),
    )

    assert summary.required_skipped == 0


def test_validate_report_rejects_missing_execution_and_required_skip(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="2" failures="0" errors="0" skipped="2">
          <testcase classname="suite" name="one">
            <skipped message="dependency missing" />
          </testcase>
          <testcase classname="suite" name="two" status="notrun" />
        </testsuite>
        """,
    )

    with pytest.raises(ReportGateError, match="executed=0"):
        validate_report(report, expected_total=2, min_executed=1)

    with pytest.raises(ReportGateError, match="required_skipped=2"):
        validate_report(report, expected_total=2, min_executed=0)


def test_validate_report_rejects_failures(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="1" failures="1" errors="0" skipped="0">
          <testcase classname="suite" name="fail"><failure message="broken" /></testcase>
        </testsuite>
        """,
    )

    with pytest.raises(ReportGateError, match="failed=1"):
        validate_report(report, expected_total=1, min_executed=1)


def test_validate_report_requires_named_testcases(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="1" failures="0" errors="0" skipped="0">
          <testcase classname="suite" name="other" />
        </testsuite>
        """,
    )

    with pytest.raises(ReportGateError, match="missing required testcase: required"):
        validate_report(report, required_cases=("required",))


def test_validate_report_rejects_required_testcase_that_was_skipped(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="2" failures="0" errors="0" skipped="1">
          <testcase classname="suite" name="required">
            <skipped message="dependency missing" />
          </testcase>
          <testcase classname="suite" name="other" />
        </testsuite>
        """,
    )

    with pytest.raises(ReportGateError, match="required testcase not executed: required"):
        validate_report(report, required_cases=("required",), allow_skip_regex=(r"dependency missing",))


def test_main_writes_machine_readable_summary(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        """
        <testsuite tests="1" failures="0" errors="0" skipped="0">
          <testcase classname="suite" name="pass" />
        </testsuite>
        """,
    )
    summary_path = tmp_path / "out" / "summary.json"

    assert main(["--report", str(report), "--json", str(summary_path)]) == 0
    assert '"executed": 1' in summary_path.read_text(encoding="utf-8")
