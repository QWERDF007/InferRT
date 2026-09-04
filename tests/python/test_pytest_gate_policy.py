"""Release skip-gate policy tests."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import _is_required_skip_item


class _MarkedItem:
    def __init__(self, *markers: str) -> None:
        self._markers = frozenset(markers)

    def get_closest_marker(self, name: str) -> Any | None:
        return object() if name in self._markers else None


@pytest.mark.parametrize(
    ("markers", "expected"),
    (
        (("integration",), True),
        (("slow",), True),
        (("integration", "optional_resource"), False),
        (("slow", "optional_resource"), False),
        (("integration", "optional_backend"), False),
        (("slow", "optional_backend"), False),
        (tuple(), False),
    ),
)
def test_required_skip_policy_excludes_only_explicit_optional_resources(
    markers: tuple[str, ...], expected: bool
) -> None:
    assert _is_required_skip_item(_MarkedItem(*markers)) is expected
