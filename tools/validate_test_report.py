"""Validate a CTest or pytest JUnit report for release CI.

The validator treats every skipped test as unverified unless its reason
matches an explicitly configured allow-list.  This keeps a successful test
runner exit code from hiding an empty or partially skipped test layer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TestReportSummary:
    report: str
    total: int
    executed: int
    passed: int
    failed: int
    errors: int
    skipped: int
    disabled: int
    required_skipped: int
    skip_reasons: tuple[str, ...]


class ReportGateError(RuntimeError):
    """Raised when a test report does not satisfy the configured gate."""


def _case_text(element: ET.Element) -> str:
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _skip_reason(testcase: ET.Element) -> str:
    skipped = testcase.find("skipped")
    if skipped is None:
        return "unspecified skip"
    message = skipped.attrib.get("message", "").strip()
    if message:
        return message
    text = _case_text(skipped)
    return text or "unspecified skip"


def _is_skipped(testcase: ET.Element) -> bool:
    status = testcase.attrib.get("status", "").strip().lower()
    return testcase.find("skipped") is not None or status in {"skip", "skipped", "notrun"}


def _is_disabled(testcase: ET.Element) -> bool:
    status = testcase.attrib.get("status", "").strip().lower()
    return testcase.find("disabled") is not None or status == "disabled"


def parse_junit(report: Path) -> TestReportSummary:
    """Parse a JUnit XML report and return per-case execution statistics."""

    report = report.resolve()
    if not report.is_file():
        raise ReportGateError(f"JUnit report does not exist: {report}")

    try:
        root = ET.parse(report).getroot()
    except (ET.ParseError, OSError) as error:
        raise ReportGateError(f"Cannot parse JUnit report {report}: {error}") from error

    testcases = list(root.iter("testcase"))
    declared_total = root.attrib.get("tests")
    if not testcases and declared_total is not None and int(declared_total) > 0:
        raise ReportGateError(f"JUnit report declares tests but contains no testcase elements: {report}")

    total = len(testcases)
    skipped_cases = [testcase for testcase in testcases if _is_skipped(testcase)]
    disabled_cases = [testcase for testcase in testcases if _is_disabled(testcase) and not _is_skipped(testcase)]
    failure_cases = [
        testcase
        for testcase in testcases
        if not _is_skipped(testcase) and testcase.find("failure") is not None
    ]
    error_cases = [
        testcase
        for testcase in testcases
        if not _is_skipped(testcase) and testcase.find("error") is not None
    ]

    skipped = len(skipped_cases)
    disabled = len(disabled_cases)
    failed = len(failure_cases)
    errors = len(error_cases)
    executed = total - skipped - disabled
    passed = executed - failed - errors
    if passed < 0:
        raise ReportGateError(
            f"Invalid JUnit counts in {report}: executed={executed}, failed={failed}, errors={errors}"
        )

    return TestReportSummary(
        report=str(report),
        total=total,
        executed=executed,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        disabled=disabled,
        required_skipped=skipped,
        skip_reasons=tuple(_skip_reason(testcase) for testcase in skipped_cases),
    )


def _reason_allowed(reason: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, reason, flags=re.IGNORECASE) for pattern in patterns)


def validate_report(
    report: Path,
    *,
    expected_total: int | None = None,
    min_executed: int = 1,
    allow_skip_regex: Iterable[str] = (),
    required_cases: Iterable[str] = (),
) -> TestReportSummary:
    """Validate failures, execution count, and explicit skip policy."""

    summary = parse_junit(report)
    patterns = tuple(allow_skip_regex)
    required_case_names = tuple(required_cases)
    if required_case_names:
        try:
            root = ET.parse(summary.report).getroot()
        except (ET.ParseError, OSError) as error:
            raise ReportGateError(f"Cannot parse JUnit report {summary.report}: {error}") from error
        available_cases = {testcase.attrib.get("name", ""): testcase for testcase in root.iter("testcase")}
        missing_cases = [name for name in required_case_names if name not in available_cases]
        skipped_required_cases = [
            name
            for name in required_case_names
            if name in available_cases and (_is_skipped(available_cases[name]) or _is_disabled(available_cases[name]))
        ]
    else:
        missing_cases = []
        skipped_required_cases = []

    required_reasons = tuple(reason for reason in summary.skip_reasons if not _reason_allowed(reason, patterns))
    summary = TestReportSummary(
        **{
            **asdict(summary),
            "required_skipped": len(required_reasons),
        }
    )

    problems: list[str] = []
    if expected_total is not None and summary.total != expected_total:
        problems.append(f"expected total={expected_total}, got total={summary.total}")
    if summary.executed < min_executed:
        problems.append(f"expected executed>={min_executed}, got executed={summary.executed}")
    if summary.failed:
        problems.append(f"failed={summary.failed}")
    if summary.errors:
        problems.append(f"errors={summary.errors}")
    if summary.required_skipped:
        problems.append(f"required_skipped={summary.required_skipped}")
    problems.extend(f"missing required testcase: {name}" for name in missing_cases)
    problems.extend(f"required testcase not executed: {name}" for name in skipped_required_cases)

    if problems:
        raise ReportGateError(f"JUnit report gate failed for {summary.report}: " + "; ".join(problems))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="CTest or pytest JUnit XML report")
    parser.add_argument("--expected-total", type=int, help="Require this exact number of testcase elements")
    parser.add_argument("--min-executed", type=int, default=1, help="Require at least this many executed cases")
    parser.add_argument(
        "--allow-skip-regex",
        action="append",
        default=[],
        help="Regular expression matching an explicitly allowed skip reason; repeat as needed",
    )
    parser.add_argument(
        "--require-case",
        action="append",
        default=[],
        help="Require an exact testcase name in the report; repeat as needed",
    )
    parser.add_argument("--json", dest="json_path", type=Path, help="Optional JSON summary output")
    args = parser.parse_args(argv)

    try:
        summary = validate_report(
            args.report,
            expected_total=args.expected_total,
            min_executed=args.min_executed,
            allow_skip_regex=args.allow_skip_regex,
            required_cases=args.require_case,
        )
    except ReportGateError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = asdict(summary)
    payload["skip_reasons"] = list(summary.skip_reasons)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
