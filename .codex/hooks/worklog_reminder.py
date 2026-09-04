#!/usr/bin/env python3
"""Codex Stop hook that reminds the agent to update WORKLOG.md.

Behavior:
- If the git working tree contains project changes but WORKLOG.md is untouched,
  ask Codex to continue once and update the ledger.
- If WORKLOG.md is already touched, allow the turn to end.
- Fail open on malformed input, missing git, non-git directories, or unexpected
  errors so this reminder can never wedge a Codex session.
- Honor stop_hook_active to avoid continuation loops.
- Standard library only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKLOG_NAME = "WORKLOG.md"


def emit(payload: dict[str, Any] | None = None) -> None:
    """Codex Stop hooks expect JSON on stdout for exit code 0."""
    print(json.dumps(payload or {}, ensure_ascii=False))


def run_git(cwd: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        emit()
        return 0

    # Codex sets this after a Stop hook has already continued the same turn.
    # Never block a second time, otherwise a reminder could loop forever.
    if payload.get("stop_hook_active"):
        emit()
        return 0

    try:
        # Prefer the cwd supplied by Codex, then let git resolve the repository root.
        cwd_value = payload.get("cwd")
        cwd = Path(cwd_value) if isinstance(cwd_value, str) and cwd_value else None

        root_result = run_git(cwd, "rev-parse", "--show-toplevel")
        if root_result.returncode != 0:
            emit()
            return 0

        root_text = root_result.stdout.strip()
        if not root_text:
            emit()
            return 0
        root = Path(root_text)

        status_result = run_git(root, "status", "--porcelain")
        if status_result.returncode != 0:
            emit()
            return 0

        changed: set[str] = set()
        for line in status_result.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip().strip('"')
            if " -> " in path:  # rename: old -> new
                path = path.split(" -> ", 1)[1]
            if path:
                changed.add(path)

        worklog_touched = any(Path(path).name == WORKLOG_NAME for path in changed)
        project_changed = any(Path(path).name != WORKLOG_NAME for path in changed)

        if project_changed and not worklog_touched:
            emit(
                {
                    "decision": "block",
                    "reason": (
                        "本轮结束前请同步 WORKLOG.md：当前 Git 工作树存在项目文件变更，"
                        "但 WORKLOG.md 尚未更新。请追加一条简洁日志，记录目标、当前状态、"
                        "必要的验证证据（执行了什么命令、关键结果是什么）以及下一步。"
                        "不要写流水账；如果这些变更并非本轮产生，也请根据当前任务判断是否需要记录。"
                    ),
                }
            )
            return 0

    except Exception:
        # Reminder only: any unexpected problem must not prevent the turn from ending.
        emit()
        return 0

    emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
