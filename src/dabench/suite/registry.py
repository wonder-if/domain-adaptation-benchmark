"""Registry for built-in benchmark suites."""

from __future__ import annotations

from dabench.suite.domainnet import SUITES as DOMAINNET_SUITES
from dabench.suite.office31 import SUITES as OFFICE31_SUITES
from dabench.suite.officehome import SUITES as OFFICE_HOME_SUITES
from dabench.task.base import BenchmarkSuite

_BUILTIN_SUITES: dict[str, BenchmarkSuite] = {
    suite.suite_id: suite
    for suite in (
        *OFFICE31_SUITES,
        *OFFICE_HOME_SUITES,
        *DOMAINNET_SUITES,
    )
}


def list_suites() -> list[BenchmarkSuite]:
    return list(_BUILTIN_SUITES.values())


def get_suite(suite_id: str) -> BenchmarkSuite:
    try:
        return _BUILTIN_SUITES[suite_id]
    except KeyError as exc:
        available = ", ".join(sorted(_BUILTIN_SUITES))
        raise ValueError(f"Unsupported benchmark suite: {suite_id}. Available suites: {available}") from exc
