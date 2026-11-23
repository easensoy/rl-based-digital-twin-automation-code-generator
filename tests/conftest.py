"""
Skip policy for the portable verification run (.github/workflows/portable-verification.yml).

With STRICT_SKIPS=1, any skipped test fails the session unless it carries a marker whose skip is
expected on some machines: ``vendor`` (EAE Buildtime absent) or ``platform_reference`` (this PyTorch
build rounds the recorded near-tie states differently). A missing checkpoint, a missing optional
package or any other skip then fails the run instead of silently shrinking coverage.
"""

from __future__ import annotations

import os

import pytest

ALLOWED_SKIP_MARKERS = ("vendor", "platform_reference")
_unexpected: list = []


def _strict() -> bool:
    return os.environ.get("STRICT_SKIPS") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.skipped and not any(item.get_closest_marker(m) for m in ALLOWED_SKIP_MARKERS):
        reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
        _unexpected.append(f"{item.nodeid}: {reason}")


def pytest_terminal_summary(terminalreporter):
    if _unexpected and _strict():
        terminalreporter.section("unexpected skips (STRICT_SKIPS=1)", red=True)
        for line in _unexpected:
            terminalreporter.line(line)


def pytest_sessionfinish(session, exitstatus):
    if _unexpected and _strict() and exitstatus == 0:
        session.exitstatus = 1
