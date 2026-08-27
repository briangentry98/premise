"""Read-only methodological validation for premise inventory graphs.

The historical validators in :mod:`premise.validation` grew together with the
export preparation code and therefore contain a mixture of checks and repairs.
This module is deliberately different: it only reads an :class:`InventoryStore`
and returns immutable, serialisable results.  Any inventory normalisation must
happen before calling this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import yaml

from .filesystem_constants import DATA_DIR
from .inventory_store import InventoryStore

VALIDATION_RULESET_VERSION = 1
ValidationSeverity = Literal["error", "warning"]
Applicability = Literal["applicable", "not_applicable"]


def _plain(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _activity_key(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        payload.get("name"),
        payload.get("reference product", payload.get("product")),
        payload.get("location"),
    )


def _exchange_provider_key(exchange: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        exchange.get("name"),
        exchange.get("product", exchange.get("reference product")),
        exchange.get("location"),
    )


@dataclass(frozen=True, slots=True)
class ActivitySelector:
    """Stable selector used by validation intents and suppressions."""

    name: str | None = None
    product: str | None = None
    location: str | None = None
    code: str | None = None
    name_pattern: str | None = None

    def __post_init__(self) -> None:
        if self.name_pattern is not None:
            re.compile(self.name_pattern)

    def matches(self, payload: Mapping[str, Any]) -> bool:
        exact = all(
            expected is None or payload.get(field_name) == expected
            for field_name, expected in (
                ("name", self.name),
                ("reference product", self.product),
                ("location", self.location),
                ("code", self.code),
            )
        )
        return exact and (
            self.name_pattern is None
            or re.fullmatch(self.name_pattern, str(payload.get("name", ""))) is not None
        )

    @classmethod
    def from_key(cls, key: Iterable[Any]) -> "ActivitySelector":
        name, product, location = tuple(key)
        return cls(name=name, product=product, location=location)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("name", self.name),
                ("product", self.product),
                ("location", self.location),
                ("code", self.code),
                ("name_pattern", self.name_pattern),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ValidationSuppression:
    """Narrow, reviewable exception to one validation rule."""

    rule_id: str
    selector: ActivitySelector
    versions: tuple[str, ...]
    system_models: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", tuple(self.versions))
        object.__setattr__(self, "system_models", tuple(self.system_models))
        if not self.rule_id or not self.explanation.strip():
            raise ValueError(
                "A validation suppression needs a rule ID and explanation."
            )
        if not self.selector.to_dict():
            raise ValueError(
                "A validation suppression needs a non-empty dataset selector."
            )
        if not self.versions or not self.system_models:
            raise ValueError(
                "A validation suppression needs applicable versions and system models."
            )

    def applies(
        self,
        issue: "ValidationIssue",
        *,
        version: str | None,
        system_model: str | None,
    ) -> bool:
        payload = {
            "name": issue.activity_key[0] if issue.activity_key else None,
            "reference product": issue.activity_key[1] if issue.activity_key else None,
            "location": issue.activity_key[2] if issue.activity_key else None,
            "code": issue.activity_code,
        }
        return (
            issue.rule_id == self.rule_id
            and ("*" in self.versions or str(version) in self.versions)
            and ("*" in self.system_models or system_model in self.system_models)
            and self.selector.matches(payload)
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One immutable methodological or structural validation finding."""

    rule_id: str
    severity: ValidationSeverity
    message: str
    applicability: Applicability = "applicable"
    checked_object_count: int = 0
    activity_id: int | None = None
    activity_key: tuple[Any, Any, Any] | None = None
    activity_code: str | None = None
    exchange_id: int | None = None
    expected: Any = None
    actual: Any = None
    tolerance: float | None = None
    suppressed: bool = False
    suppression_explanation: str | None = None

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("Validation issues require a stable rule ID.")
        if self.severity not in {"error", "warning"}:
            raise ValueError(f"Invalid validation severity: {self.severity!r}.")
        if self.applicability not in {"applicable", "not_applicable"}:
            raise ValueError(
                f"Invalid validation applicability: {self.applicability!r}."
            )
        object.__setattr__(self, "activity_key", _freeze(self.activity_key))
        object.__setattr__(self, "expected", _freeze(self.expected))
        object.__setattr__(self, "actual", _freeze(self.actual))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "applicability": self.applicability,
            "checked_object_count": self.checked_object_count,
            "activity_id": self.activity_id,
            "activity_key": _plain(self.activity_key),
            "activity_code": self.activity_code,
            "exchange_id": self.exchange_id,
            "expected": _plain(self.expected),
            "actual": _plain(self.actual),
            "tolerance": self.tolerance,
            "suppressed": self.suppressed,
            "suppression_explanation": self.suppression_explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationIssue":
        payload = dict(data)
        if payload.get("activity_key") is not None:
            payload["activity_key"] = tuple(payload["activity_key"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ValidationRuleResult:
    """Complete outcome of one rule, including successful zero-issue checks."""

    rule_id: str
    severity: ValidationSeverity
    applicability: Applicability
    checked_object_count: int
    expected: Any = None
    actual: Any = None
    tolerance: float | None = None
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _freeze(self.expected))
        object.__setattr__(self, "actual", _freeze(self.actual))
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "applicability": self.applicability,
            "checked_object_count": self.checked_object_count,
            "expected": _plain(self.expected),
            "actual": _plain(self.actual),
            "tolerance": self.tolerance,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationRuleResult":
        payload = dict(data)
        payload["issues"] = tuple(
            ValidationIssue.from_dict(issue) for issue in payload.get("issues", ())
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable result of a complete or targeted validation pass."""

    scenario_identity: Any
    store_generation: int
    ruleset_version: int
    certificate_key: str
    rule_results: tuple[ValidationRuleResult, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_identity", _freeze(self.scenario_identity))
        object.__setattr__(self, "rule_results", tuple(self.rule_results))

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for result in self.rule_results for issue in result.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == "error" and not issue.suppressed
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == "warning" and not issue.suppressed
        )

    @property
    def suppressed_issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.suppressed)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise PremiseValidationError(self)

    def with_reuse(self, reused: bool) -> "ValidationReport":
        return replace(self, reused=reused)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_identity": _plain(self.scenario_identity),
            "store_generation": self.store_generation,
            "ruleset_version": self.ruleset_version,
            "certificate_key": self.certificate_key,
            "rule_results": [result.to_dict() for result in self.rule_results],
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        payload = dict(data)
        payload["rule_results"] = tuple(
            ValidationRuleResult.from_dict(result)
            for result in payload.get("rule_results", ())
        )
        return cls(**payload)


class PremiseValidationError(ValueError):
    """Raised when a report contains one or more unsuppressed errors."""

    def __init__(self, report: ValidationReport):
        self.report = report

        def describe(issue: ValidationIssue) -> str:
            details = []
            if issue.activity_key is not None:
                details.append(f"activity={tuple(issue.activity_key)!r}")
            if issue.exchange_id is not None:
                details.append(f"exchange_id={issue.exchange_id}")
            if issue.expected is not None:
                expected = repr(_plain(issue.expected))
                if len(expected) > 120:
                    expected = f"{expected[:117]}..."
                details.append(f"expected={expected}")
            if issue.actual is not None:
                actual = repr(_plain(issue.actual))
                if len(actual) > 160:
                    actual = f"{actual[:157]}..."
                details.append(f"actual={actual}")
            detail = f" ({', '.join(details)})" if details else ""
            return f"{issue.rule_id}: {issue.message}{detail}"

        error_counts: dict[str, int] = defaultdict(int)
        representative_issues: dict[str, list[ValidationIssue]] = defaultdict(list)
        for issue in report.errors:
            error_counts[issue.rule_id] += 1
            limit = 5 if issue.rule_id == "GRAPH.NEW_FORBIDDEN_CYCLE" else 1
            if len(representative_issues[issue.rule_id]) < limit:
                representative_issues[issue.rule_id].append(issue)
        preview = "; ".join(
            describe(issue)
            for _, issues in sorted(representative_issues.items())
            for issue in issues
        )
        shown = sum(len(issues) for issues in representative_issues.values())
        suffix = "" if len(report.errors) <= shown else "; ..."
        counts = ", ".join(
            f"{rule_id}={count}" for rule_id, count in sorted(error_counts.items())
        )
        super().__init__(
            f"Inventory validation failed with {len(report.errors)} "
            f"unsuppressed error(s) [{counts}]: {preview}{suffix}"
        )


@dataclass(frozen=True, slots=True)
class ValidationIntent:
    """Targets and independent expectations declared by a transformation.

    Keys are semantic activity triples ``(name, reference product, location)``.
    Supplier vectors map a target activity key to ``((supplier_key, share), ...)``.
    ``baseline_fingerprints`` enables collateral-mutation detection without
    keeping a mutable copy of the source graph.
    """

    transformation: str
    affected_activity_ids: frozenset[int] = frozenset()
    affected_activity_keys: frozenset[tuple[Any, Any, Any]] = frozenset()
    expected_match_count: int | None = None
    expected_regions: tuple[str, ...] = ()
    expected_technologies: tuple[str, ...] = ()
    algorithm: str | None = None
    intended_suppliers: Mapping[
        tuple[Any, Any, Any],
        tuple[tuple[tuple[Any, Any, Any], float], ...],
    ] = field(default_factory=dict)
    computed_target_values: Mapping[str, Any] = field(default_factory=dict)
    baseline_fingerprints: Mapping[tuple[Any, Any, Any], str] = field(
        default_factory=dict
    )
    allowed_added_keys: frozenset[tuple[Any, Any, Any]] = frozenset()
    allowed_removed_keys: frozenset[tuple[Any, Any, Any]] = frozenset()
    baseline_cycles: frozenset[frozenset[tuple[Any, Any, Any]]] = frozenset()
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "affected_activity_ids", frozenset(self.affected_activity_ids)
        )
        object.__setattr__(
            self, "affected_activity_keys", frozenset(self.affected_activity_keys)
        )
        object.__setattr__(self, "expected_regions", tuple(self.expected_regions))
        object.__setattr__(
            self, "expected_technologies", tuple(self.expected_technologies)
        )
        object.__setattr__(self, "intended_suppliers", _freeze(self.intended_suppliers))
        object.__setattr__(
            self, "computed_target_values", _freeze(self.computed_target_values)
        )
        object.__setattr__(
            self, "baseline_fingerprints", _freeze(self.baseline_fingerprints)
        )
        object.__setattr__(
            self, "allowed_added_keys", frozenset(self.allowed_added_keys)
        )
        object.__setattr__(
            self, "allowed_removed_keys", frozenset(self.allowed_removed_keys)
        )
        object.__setattr__(
            self,
            "baseline_cycles",
            frozenset(frozenset(cycle) for cycle in self.baseline_cycles),
        )


@dataclass(frozen=True, slots=True)
class ValidationCertificate:
    """Cacheable proof that one exact store generation passed validation."""

    cache_key: str
    store_generation: int
    ruleset_version: int
    scenario_identity: Any
    source_fingerprint: str
    iam_fingerprint: str
    system_model: str
    version: str
    report: ValidationReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_identity", _freeze(self.scenario_identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "store_generation": self.store_generation,
            "ruleset_version": self.ruleset_version,
            "scenario_identity": _plain(self.scenario_identity),
            "source_fingerprint": self.source_fingerprint,
            "iam_fingerprint": self.iam_fingerprint,
            "system_model": self.system_model,
            "version": self.version,
            "report": self.report.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationCertificate":
        payload = dict(data)
        payload["report"] = ValidationReport.from_dict(payload["report"])
        return cls(**payload)


RULES: tuple[tuple[str, ValidationSeverity], ...] = (
    ("GRAPH.REQUIRED_ACTIVITY_FIELDS", "error"),
    ("GRAPH.EXCHANGE_TYPE", "error"),
    ("GRAPH.FINITE_NUMERIC", "error"),
    ("GRAPH.UNCERTAINTY", "error"),
    ("GRAPH.PRODUCTION_REFERENCE", "error"),
    ("GRAPH.PROVIDER_EXISTS", "error"),
    ("GRAPH.PROVIDER_AMBIGUOUS", "error"),
    ("GRAPH.PROVIDER_PRODUCT_UNIT", "error"),
    ("GRAPH.GEOGRAPHIC_FALLBACK", "error"),
    ("GRAPH.NEGATIVE_MARKET_SHARE", "error"),
    ("GRAPH.DUPLICATE_SUPPLIER", "error"),
    ("GRAPH.TRANSFORMATION_SCOPE", "error"),
    ("GRAPH.RULE_TARGET_CARDINALITY", "error"),
    ("GRAPH.NEW_FORBIDDEN_CYCLE", "error"),
    ("METHOD.SUPPLIER_VECTOR", "error"),
    ("METHOD.CONSEQUENTIAL_ALGORITHM", "error"),
    ("METHOD.EXPECTED_COVERAGE", "error"),
)


@dataclass(slots=True)
class _Accumulator:
    rule_id: str
    severity: ValidationSeverity
    applicability: Applicability = "applicable"
    checked: int = 0
    expected: Any = None
    tolerance: float | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    def issue(self, message: str, **kwargs: Any) -> None:
        self.issues.append(
            ValidationIssue(
                rule_id=self.rule_id,
                severity=self.severity,
                message=message,
                expected=kwargs.pop("expected", self.expected),
                tolerance=kwargs.pop("tolerance", self.tolerance),
                **kwargs,
            )
        )

    def result(self) -> ValidationRuleResult:
        issues = tuple(
            replace(issue, checked_object_count=self.checked) for issue in self.issues
        )
        return ValidationRuleResult(
            rule_id=self.rule_id,
            severity=self.severity,
            applicability=self.applicability,
            checked_object_count=self.checked,
            expected=self.expected,
            actual={"issue_count": len(issues)},
            tolerance=self.tolerance,
            issues=issues,
        )


def _iter_storage(store: InventoryStore):
    """Yield activity metadata and exchange storage without graph materialisation."""

    underlying = getattr(store, "_store", store)
    iterator = getattr(underlying, "_iter_storage_activities", None)
    exchange_getter = getattr(underlying, "_storage_exchange", None)
    if iterator is not None and exchange_getter is not None:
        for activity_id, payload, exchange_ids in iterator():
            yield activity_id, payload, tuple(
                (exchange_id, exchange_getter(exchange_id))
                for exchange_id in exchange_ids
            )
        return

    for record in store.iter_activities():
        yield record.id, record, tuple(
            (exchange_id, store.exchange(exchange_id))
            for exchange_id in record.exchange_ids
        )


def _iter_activity_metadata(store: InventoryStore):
    """Yield activity metadata without constructing exchange views."""

    underlying = getattr(store, "_store", store)
    iterator = getattr(underlying, "_iter_storage_activities", None)
    if iterator is not None:
        for activity_id, payload, _ in iterator():
            yield activity_id, payload
        return
    for record in store.iter_activities():
        yield record.id, record


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    try:
        return _numeric(value) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _payload_fingerprint(
    payload: Mapping[str, Any], exchanges: Iterable[tuple[int, Mapping[str, Any]]]
) -> str:
    activity = dict(payload)
    activity["exchanges"] = [dict(exchange) for _, exchange in exchanges]
    return _stable_hash(activity)


def inventory_store_fingerprint(store: InventoryStore) -> str:
    """Hash a store deterministically without building a materialised database list."""

    digest = hashlib.sha256()
    for activity_id, payload, exchanges in _iter_storage(store):
        digest.update(str(activity_id).encode("ascii"))
        digest.update(_payload_fingerprint(payload, exchanges).encode("ascii"))
    return digest.hexdigest()


def inventory_activity_fingerprints(
    store: InventoryStore,
) -> Mapping[tuple[Any, Any, Any], str]:
    """Return immutable per-activity hashes for transformation-scope contracts."""

    return MappingProxyType(
        {
            _activity_key(payload): _payload_fingerprint(payload, exchanges)
            for _, payload, exchanges in _iter_storage(store)
        }
    )


def _cycle_signatures(
    adjacency: Mapping[int, set[int]],
    keys_by_id: Mapping[int, tuple[Any, Any, Any]],
) -> frozenset[frozenset[tuple[Any, Any, Any]]]:
    """Return strongly-connected components without recursive graph walking."""

    reverse: dict[int, set[int]] = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)

    visited: set[int] = set()
    order: list[int] = []
    for root in keys_by_id:
        if root in visited:
            continue
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend(
                (neighbour, False)
                for neighbour in adjacency.get(node, ())
                if neighbour not in visited
            )

    cycles: set[frozenset[tuple[Any, Any, Any]]] = set()
    assigned: set[int] = set()
    for root in reversed(order):
        if root in assigned:
            continue
        component = []
        stack = [root]
        assigned.add(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in reverse.get(node, ()):
                if neighbour not in assigned:
                    assigned.add(neighbour)
                    stack.append(neighbour)
        if len(component) > 1 or root in adjacency.get(root, ()):
            cycles.add(frozenset(keys_by_id[member] for member in component))
    return frozenset(cycles)


def inventory_cycle_signatures(
    store: InventoryStore,
) -> frozenset[frozenset[tuple[Any, Any, Any]]]:
    """Return semantic cycle signatures for a baseline inventory graph."""

    keys_by_id: dict[int, tuple[Any, Any, Any]] = {}
    ids_by_key: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    ids_by_identifier: dict[tuple[Any, Any], int] = {}
    for activity_id, payload in _iter_activity_metadata(store):
        key = _activity_key(payload)
        keys_by_id[activity_id] = key
        ids_by_key[key].append(activity_id)
        if payload.get("database") is not None and payload.get("code") is not None:
            ids_by_identifier[(payload["database"], payload["code"])] = activity_id

    adjacency: dict[int, set[int]] = defaultdict(set)
    for activity_id, _, exchanges in _iter_storage(store):
        for _, exchange in exchanges:
            if exchange.get("type") != "technosphere":
                continue
            provider_id = None
            exchange_input = exchange.get("input")
            if isinstance(exchange_input, (tuple, list)) and len(exchange_input) == 2:
                provider_id = ids_by_identifier.get(tuple(exchange_input))
            if provider_id is None:
                matches = ids_by_key.get(_exchange_provider_key(exchange), ())
                if len(matches) == 1:
                    provider_id = matches[0]
            if provider_id is not None:
                adjacency[activity_id].add(provider_id)
    return _cycle_signatures(adjacency, keys_by_id)


def _load_suppressions(path: Path) -> tuple[ValidationSuppression, ...]:
    if not path.exists():
        return ()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError("Validation suppressions must be a YAML list.")
    suppressions = []
    for row in data:
        if not isinstance(row, Mapping):
            raise ValueError("Each validation suppression must be a mapping.")
        suppressions.append(
            ValidationSuppression(
                rule_id=row.get("rule_id", ""),
                selector=ActivitySelector(**dict(row.get("dataset", {}))),
                versions=tuple(str(item) for item in row.get("versions", ())),
                system_models=tuple(row.get("system_models", ())),
                explanation=row.get("explanation", ""),
            )
        )
    return tuple(suppressions)


@lru_cache(maxsize=1)
def load_validation_suppressions() -> tuple[ValidationSuppression, ...]:
    return _load_suppressions(DATA_DIR / "utils" / "validation" / "suppressions.yaml")


class InventoryGraphValidator:
    """Single-pass, read-only validator for an inventory-store generation."""

    valid_exchange_types = frozenset({"production", "technosphere", "biosphere"})
    required_activity_fields = ("name", "reference product", "location", "unit")
    uncertainty_fields = {
        2: ("loc", "scale"),
        3: ("loc", "scale"),
        4: ("minimum", "maximum"),
        5: ("loc", "minimum", "maximum"),
        6: ("loc", "minimum", "maximum"),
        7: ("minimum", "maximum"),
        8: ("loc", "scale", "shape"),
        9: ("loc", "scale", "shape"),
        10: ("loc", "scale", "shape"),
        11: ("loc", "scale", "shape"),
        12: ("loc", "scale", "shape"),
    }

    def __init__(
        self,
        store: InventoryStore,
        *,
        scenario_identity: Any = None,
        source_fingerprint: str = "unknown",
        iam_fingerprint: str = "unknown",
        system_model: str = "cutoff",
        version: str = "unknown",
        intent: ValidationIntent | None = None,
        baseline_cycles: Iterable[Iterable[tuple[Any, Any, Any]]] = (),
        suppressions: Iterable[ValidationSuppression] | None = None,
    ) -> None:
        self.store = store
        self.scenario_identity = (
            scenario_identity
            if scenario_identity is not None
            else getattr(store, "scenario_identity", None)
        )
        self.source_fingerprint = str(source_fingerprint)
        self.iam_fingerprint = str(iam_fingerprint)
        self.system_model = str(system_model)
        self.version = str(version)
        self.intent = intent
        self.baseline_cycles = frozenset(
            frozenset(tuple(key) for key in cycle) for cycle in baseline_cycles
        )
        self.suppressions = tuple(
            load_validation_suppressions() if suppressions is None else suppressions
        )
        self.generation = int(getattr(store, "generation", 0))
        self.cache_key = validation_cache_key(
            store_generation=self.generation,
            scenario_identity=self.scenario_identity,
            source_fingerprint=self.source_fingerprint,
            iam_fingerprint=self.iam_fingerprint,
            system_model=self.system_model,
            version=self.version,
            intent=self.intent,
            baseline_cycles=self.baseline_cycles,
        )

    def _cached_certificate(self) -> ValidationCertificate | None:
        underlying = getattr(self.store, "_store", self.store)
        payload = getattr(underlying, "_validation_certificate_payload", None)
        if not isinstance(payload, Mapping):
            return None
        try:
            certificate = ValidationCertificate.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            certificate.cache_key == self.cache_key
            and certificate.ruleset_version == VALIDATION_RULESET_VERSION
            and certificate.store_generation == self.generation
        ):
            return certificate
        return None

    def certify(self, *, raise_on_error: bool = True) -> ValidationCertificate:
        cached = self._cached_certificate()
        if cached is not None:
            report = cached.report.with_reuse(True)
            if raise_on_error:
                report.raise_for_errors()
            return replace(cached, report=report)

        report = self.validate()
        certificate = ValidationCertificate(
            cache_key=self.cache_key,
            store_generation=self.generation,
            ruleset_version=VALIDATION_RULESET_VERSION,
            scenario_identity=self.scenario_identity,
            source_fingerprint=self.source_fingerprint,
            iam_fingerprint=self.iam_fingerprint,
            system_model=self.system_model,
            version=self.version,
            report=report,
        )
        underlying = getattr(self.store, "_store", self.store)
        underlying._validation_certificate_payload = certificate.to_dict()
        if raise_on_error:
            report.raise_for_errors()
        return certificate

    def validate(self) -> ValidationReport:
        rules = {
            rule_id: _Accumulator(rule_id, severity) for rule_id, severity in RULES
        }
        keys_by_id: dict[int, tuple[Any, Any, Any]] = {}
        activity_payloads: dict[int, Mapping[str, Any]] = {}
        ids_by_key: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
        ids_by_name_location: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        ids_by_identifier: dict[tuple[Any, Any], int] = {}

        # Metadata and exchange payloads are retained as immutable store views;
        # no activity or exchange dictionaries are materialised here.
        for activity_id, payload in _iter_activity_metadata(self.store):
            key = _activity_key(payload)
            keys_by_id[activity_id] = key
            activity_payloads[activity_id] = payload
            ids_by_key[key].append(activity_id)
            ids_by_name_location[(key[0], key[2])].append(activity_id)
            if payload.get("database") is not None and payload.get("code") is not None:
                ids_by_identifier[(payload["database"], payload["code"])] = activity_id

        intent = self.intent
        target_ids = set(keys_by_id)
        if intent is not None and (
            intent.affected_activity_ids or intent.affected_activity_keys
        ):
            target_ids = set(intent.affected_activity_ids)
            target_ids.update(
                activity_id
                for key in intent.affected_activity_keys
                for activity_id in ids_by_key.get(tuple(key), ())
            )

        cardinality = rules["GRAPH.RULE_TARGET_CARDINALITY"]
        if intent is None:
            cardinality.applicability = "not_applicable"
        else:
            cardinality.checked = len(target_ids)
            expected_count = intent.expected_match_count
            if expected_count is None:
                expected_count = len(intent.affected_activity_ids) + len(
                    intent.affected_activity_keys
                )
            cardinality.expected = expected_count
            if expected_count != len(target_ids):
                cardinality.issue(
                    "Validation targets did not resolve to the expected number of activities.",
                    expected=expected_count,
                    actual=len(target_ids),
                )

        adjacency: dict[int, set[int]] = defaultdict(set)
        fingerprints: dict[tuple[Any, Any, Any], str] = {}
        actual_vectors: dict[
            tuple[Any, Any, Any], dict[tuple[Any, Any, Any], float]
        ] = {}

        for activity_id, payload, exchanges in _iter_storage(self.store):
            key = keys_by_id[activity_id]
            if intent is not None and intent.baseline_fingerprints:
                fingerprints[key] = _payload_fingerprint(payload, exchanges)
            if activity_id not in target_ids:
                continue

            required = rules["GRAPH.REQUIRED_ACTIVITY_FIELDS"]
            required.checked += 1
            missing = [
                field_name
                for field_name in self.required_activity_fields
                if not payload.get(field_name)
            ]
            if missing:
                required.issue(
                    "Activity is missing required fields.",
                    activity_id=activity_id,
                    activity_key=key,
                    activity_code=payload.get("code"),
                    expected=self.required_activity_fields,
                    actual=missing,
                )

            reference_productions = 0
            negative_reference_product = any(
                exchange.get("type") == "production"
                and (
                    exchange.get("name", key[0]),
                    exchange.get("product", exchange.get("reference product", key[1])),
                    exchange.get("location", key[2]),
                )
                == key
                and _finite(exchange.get("amount"))
                and float(exchange["amount"]) < 0
                for _, exchange in exchanges
            )
            seen_suppliers: dict[str, int] = {}
            vector: dict[tuple[Any, Any, Any], float] = defaultdict(float)
            for exchange_id, exchange in exchanges:
                exchange_type = exchange.get("type")
                exchange_type_rule = rules["GRAPH.EXCHANGE_TYPE"]
                exchange_type_rule.checked += 1
                if exchange_type not in self.valid_exchange_types:
                    exchange_type_rule.issue(
                        "Exchange has an invalid or missing type.",
                        activity_id=activity_id,
                        activity_key=key,
                        activity_code=payload.get("code"),
                        exchange_id=exchange_id,
                        expected=tuple(sorted(self.valid_exchange_types)),
                        actual=exchange_type,
                    )

                finite_rule = rules["GRAPH.FINITE_NUMERIC"]
                finite_rule.checked += 1
                if not _finite(exchange.get("amount")):
                    finite_rule.issue(
                        "Exchange amount must be a finite numeric value.",
                        activity_id=activity_id,
                        activity_key=key,
                        activity_code=payload.get("code"),
                        exchange_id=exchange_id,
                        expected="finite numeric amount",
                        actual=exchange.get("amount"),
                    )

                self._check_uncertainty(
                    rules["GRAPH.UNCERTAINTY"],
                    activity_id,
                    key,
                    payload,
                    exchange_id,
                    exchange,
                )

                if exchange_type == "production":
                    production = rules["GRAPH.PRODUCTION_REFERENCE"]
                    production.checked += 1
                    exchange_key = (
                        exchange.get("name", key[0]),
                        exchange.get(
                            "product", exchange.get("reference product", key[1])
                        ),
                        exchange.get("location", key[2]),
                    )
                    if exchange_key == key and exchange.get(
                        "unit", payload.get("unit")
                    ) == payload.get("unit"):
                        reference_productions += 1
                    continue

                if exchange_type != "technosphere":
                    continue

                provider_key = _exchange_provider_key(exchange)
                vector[provider_key] += float(exchange.get("amount", 0) or 0)
                provider_id = None
                exchange_input = exchange.get("input")
                if (
                    isinstance(exchange_input, (tuple, list))
                    and len(exchange_input) == 2
                ):
                    provider_id = ids_by_identifier.get(tuple(exchange_input))
                matches = ids_by_key.get(provider_key, ())

                exists = rules["GRAPH.PROVIDER_EXISTS"]
                exists.checked += 1
                if provider_id is None and not matches:
                    exists.issue(
                        "Technosphere exchange links to no activity in the inventory graph.",
                        activity_id=activity_id,
                        activity_key=key,
                        activity_code=payload.get("code"),
                        exchange_id=exchange_id,
                        expected="one provider",
                        actual=0,
                    )

                ambiguous = rules["GRAPH.PROVIDER_AMBIGUOUS"]
                ambiguous.checked += 1
                if provider_id is None and len(matches) > 1:
                    ambiguous.issue(
                        "Technosphere exchange has multiple providers and no resolving input identifier.",
                        activity_id=activity_id,
                        activity_key=key,
                        activity_code=payload.get("code"),
                        exchange_id=exchange_id,
                        expected=1,
                        actual=len(matches),
                    )

                agreement = rules["GRAPH.PROVIDER_PRODUCT_UNIT"]
                agreement.checked += 1
                resolved_id = (
                    provider_id
                    if provider_id is not None
                    else (matches[0] if len(matches) == 1 else None)
                )
                if resolved_id is not None:
                    provider = activity_payloads[resolved_id]
                    expected_product = provider.get(
                        "reference product", provider.get("product")
                    )
                    expected_unit = provider.get("unit")
                    if (
                        exchange.get("name") != provider.get("name")
                        or exchange.get("location") != provider.get("location")
                        or exchange.get("product") != expected_product
                        or exchange.get("unit") != expected_unit
                    ):
                        agreement.issue(
                            "Technosphere exchange product or unit disagrees with its provider.",
                            activity_id=activity_id,
                            activity_key=key,
                            activity_code=payload.get("code"),
                            exchange_id=exchange_id,
                            expected={
                                "product": expected_product,
                                "unit": expected_unit,
                                "name": provider.get("name"),
                                "location": provider.get("location"),
                            },
                            actual={
                                "product": exchange.get("product"),
                                "unit": exchange.get("unit"),
                                "name": exchange.get("name"),
                                "location": exchange.get("location"),
                            },
                        )
                    adjacency[activity_id].add(resolved_id)
                elif not matches:
                    name_location_matches = ids_by_name_location.get(
                        (exchange.get("name"), exchange.get("location")), ()
                    )
                    if name_location_matches:
                        agreement.issue(
                            "Technosphere provider name/location exists but product does not match.",
                            activity_id=activity_id,
                            activity_key=key,
                            activity_code=payload.get("code"),
                            exchange_id=exchange_id,
                            expected=sorted(
                                {
                                    activity_payloads[item].get("reference product")
                                    for item in name_location_matches
                                }
                            ),
                            actual=exchange.get("product"),
                        )

                expected_entries = (
                    intent.intended_suppliers.get(key, ()) if intent is not None else ()
                )
                if expected_entries:
                    fallback = rules["GRAPH.GEOGRAPHIC_FALLBACK"]
                    fallback.checked += 1
                    self._check_geographic_fallback(
                        fallback,
                        activity_id,
                        key,
                        payload,
                        exchange_id,
                        exchange,
                        expected_entries,
                    )

                market_share = rules["GRAPH.NEGATIVE_MARKET_SHARE"]
                if self._is_market_share(payload, exchange, negative_reference_product):
                    market_share.checked += 1
                    if (
                        _finite(exchange.get("amount"))
                        and float(exchange["amount"]) < 0
                    ):
                        market_share.issue(
                            "A market supplier share is negative.",
                            activity_id=activity_id,
                            activity_key=key,
                            activity_code=payload.get("code"),
                            exchange_id=exchange_id,
                            expected=">= 0",
                            actual=exchange.get("amount"),
                        )

                duplicate = rules["GRAPH.DUPLICATE_SUPPLIER"]
                duplicate.checked += 1
                # Providers can legitimately share a semantic activity key when
                # an ``input`` code disambiguates them. Only byte-for-byte
                # equivalent exchange records are accidental duplicates.
                supplier_signature = _stable_hash(dict(exchange))
                if supplier_signature in seen_suppliers:
                    duplicate.issue(
                        "Activity contains duplicate exchanges to the same supplier.",
                        activity_id=activity_id,
                        activity_key=key,
                        activity_code=payload.get("code"),
                        exchange_id=exchange_id,
                        expected="one supplier exchange",
                        actual={
                            "first_exchange_id": seen_suppliers[supplier_signature],
                            "supplier": _exchange_provider_key(exchange),
                            "amount": exchange.get("amount"),
                        },
                    )
                else:
                    seen_suppliers[supplier_signature] = exchange_id

            production = rules["GRAPH.PRODUCTION_REFERENCE"]
            if reference_productions != 1:
                production.issue(
                    "Activity must have exactly one matching reference-production exchange.",
                    activity_id=activity_id,
                    activity_key=key,
                    activity_code=payload.get("code"),
                    expected=1,
                    actual=reference_productions,
                )
            actual_vectors[key] = dict(vector)

        self._check_scope(rules["GRAPH.TRANSFORMATION_SCOPE"], keys_by_id, fingerprints)
        self._check_supplier_vectors(rules["METHOD.SUPPLIER_VECTOR"], actual_vectors)
        self._check_algorithm(rules["METHOD.CONSEQUENTIAL_ALGORITHM"])
        self._check_coverage(rules["METHOD.EXPECTED_COVERAGE"], keys_by_id)
        if rules["GRAPH.GEOGRAPHIC_FALLBACK"].checked == 0:
            rules["GRAPH.GEOGRAPHIC_FALLBACK"].applicability = "not_applicable"

        cycles_rule = rules["GRAPH.NEW_FORBIDDEN_CYCLE"]
        cycles_rule.checked = sum(len(values) for values in adjacency.values())
        cycles = _cycle_signatures(adjacency, keys_by_id)
        baseline_cycles = (
            intent.baseline_cycles
            if intent is not None and intent.baseline_cycles
            else self.baseline_cycles
        )

        def cycle_shape(cycle):
            return frozenset(
                (
                    re.sub(r", \d+-year period$", "", str(key[0])),
                    key[1],
                )
                for key in cycle
            )

        baseline_cycle_shapes = {cycle_shape(cycle) for cycle in baseline_cycles}
        for cycle in cycles.difference(baseline_cycles):
            # Regional proxy creation retains the method structure of a source
            # cycle while changing activity locations. Such a cycle is baseline
            # lineage, not a newly introduced feedback loop.
            shaped_cycle = cycle_shape(cycle)
            if any(
                baseline_cycle.issubset(cycle) for baseline_cycle in baseline_cycles
            ) or any(
                baseline_shape.issubset(shaped_cycle)
                for baseline_shape in baseline_cycle_shapes
            ):
                continue
            representative = min(cycle, key=repr, default=None)
            cycles_rule.issue(
                "Transformation introduced a technosphere cycle absent from the baseline.",
                activity_key=representative,
                expected="baseline cycle or acyclic graph",
                actual=sorted(cycle, key=repr),
            )

        results = tuple(
            self._apply_suppressions(rule.result()) for rule in rules.values()
        )
        return ValidationReport(
            scenario_identity=self.scenario_identity,
            store_generation=self.generation,
            ruleset_version=VALIDATION_RULESET_VERSION,
            certificate_key=self.cache_key,
            rule_results=results,
        )

    def _check_uncertainty(
        self,
        rule: _Accumulator,
        activity_id: int,
        activity_key: tuple[Any, Any, Any],
        activity: Mapping[str, Any],
        exchange_id: int,
        exchange: Mapping[str, Any],
    ) -> None:
        rule.checked += 1
        uncertainty_type = exchange.get("uncertainty type", 0)
        try:
            uncertainty_type = int(uncertainty_type)
        except (TypeError, ValueError, OverflowError):
            uncertainty_type = -1
        common = {
            "activity_id": activity_id,
            "activity_key": activity_key,
            "activity_code": activity.get("code"),
            "exchange_id": exchange_id,
        }
        if uncertainty_type not in range(13):
            rule.issue(
                "Exchange has an unsupported uncertainty type.",
                expected="integer from 0 to 12",
                actual=exchange.get("uncertainty type"),
                **common,
            )
            return
        required = self.uncertainty_fields.get(uncertainty_type, ())
        missing = [field_name for field_name in required if field_name not in exchange]
        if missing:
            rule.issue(
                "Exchange uncertainty parameters are incomplete.",
                expected=required,
                actual={"missing": missing},
                **common,
            )
            return
        invalid = {
            field_name: exchange.get(field_name)
            for field_name in required
            if not _finite(exchange.get(field_name))
        }
        if invalid:
            rule.issue(
                "Exchange uncertainty parameters must be finite numbers.",
                expected="finite numeric parameters",
                actual=invalid,
                **common,
            )
            return
        if "scale" in required and float(exchange["scale"]) < 0:
            rule.issue(
                "Uncertainty scale must not be negative.",
                expected=">= 0",
                actual=exchange["scale"],
                **common,
            )
        if "minimum" in required and "maximum" in required:
            minimum = float(exchange["minimum"])
            maximum = float(exchange["maximum"])
            if minimum > maximum:
                rule.issue(
                    "Uncertainty minimum exceeds maximum.",
                    expected="minimum <= maximum",
                    actual={"minimum": minimum, "maximum": maximum},
                    **common,
                )
            if "loc" in required and not minimum <= float(exchange["loc"]) <= maximum:
                rule.issue(
                    "Uncertainty location lies outside its bounds.",
                    expected={"minimum": minimum, "maximum": maximum},
                    actual=exchange["loc"],
                    **common,
                )
        if uncertainty_type == 2:
            amount = exchange.get("amount")
            if (
                _finite(amount)
                and float(amount) < 0
                and not exchange.get("negative", False)
            ):
                rule.issue(
                    "Negative lognormal amount is missing the negative-sign convention.",
                    expected={"negative": True},
                    actual={"negative": exchange.get("negative", False)},
                    **common,
                )

    @staticmethod
    def _is_market_share(
        activity: Mapping[str, Any],
        exchange: Mapping[str, Any],
        negative_reference_product: bool = False,
    ) -> bool:
        name = str(activity.get("name", "")).lower()
        product = str(activity.get("reference product", "")).lower()
        supplier = str(exchange.get("name", "")).lower()
        waste_tokens = (
            "waste",
            "treatment",
            "scrap",
            "residue",
            "sewage",
            "sludge",
        )
        return (
            (name.startswith("market for ") or name.startswith("market group for "))
            and not negative_reference_product
            and exchange.get("unit") == activity.get("unit")
            and exchange.get("product")
            == activity.get("reference product", activity.get("product"))
            and not any(token in name or token in product for token in waste_tokens)
            and not any(token in supplier for token in waste_tokens)
        )

    @staticmethod
    def _check_geographic_fallback(
        rule: _Accumulator,
        activity_id: int,
        activity_key: tuple[Any, Any, Any],
        activity: Mapping[str, Any],
        exchange_id: int,
        exchange: Mapping[str, Any],
        expected_entries: Iterable[tuple[tuple[Any, Any, Any], float]],
    ) -> None:
        linked_location = exchange.get("location")
        expected_locations = {
            supplier_key[2]
            for supplier_key, _ in expected_entries
            if supplier_key[0] == exchange.get("name")
            and supplier_key[1] == exchange.get("product")
        }
        if expected_locations and linked_location not in expected_locations:
            rule.issue(
                "Technosphere exchange uses a lower-ranked geographic fallback while a better provider exists.",
                activity_id=activity_id,
                activity_key=activity_key,
                activity_code=activity.get("code"),
                exchange_id=exchange_id,
                expected={
                    "locations": sorted(expected_locations),
                },
                actual={"location": linked_location},
            )

    def _check_scope(
        self,
        rule: _Accumulator,
        keys_by_id: Mapping[int, tuple[Any, Any, Any]],
        fingerprints: Mapping[tuple[Any, Any, Any], str],
    ) -> None:
        intent = self.intent
        if intent is None or not intent.baseline_fingerprints:
            rule.applicability = "not_applicable"
            return
        current_keys = set(keys_by_id.values())
        baseline_keys = set(intent.baseline_fingerprints)
        targeted = set(intent.affected_activity_keys)
        targeted.update(
            keys_by_id.get(activity_id) for activity_id in intent.affected_activity_ids
        )
        targeted.discard(None)
        rule.checked = len(baseline_keys | current_keys)
        unexpected_added = current_keys - baseline_keys - set(intent.allowed_added_keys)
        unexpected_removed = (
            baseline_keys - current_keys - set(intent.allowed_removed_keys)
        )
        for key in sorted(unexpected_added, key=repr):
            rule.issue(
                "Transformation added an activity outside its declared scope.",
                activity_key=key,
                expected="declared addition",
                actual="unexpected addition",
            )
        for key in sorted(unexpected_removed, key=repr):
            rule.issue(
                "Transformation removed an activity outside its declared scope.",
                activity_key=key,
                expected="declared removal",
                actual="unexpected removal",
            )
        for key in sorted(baseline_keys & current_keys - targeted, key=repr):
            if fingerprints.get(key) != intent.baseline_fingerprints[key]:
                rule.issue(
                    "Transformation modified an activity outside its declared targets.",
                    activity_key=key,
                    expected=intent.baseline_fingerprints[key],
                    actual=fingerprints.get(key),
                )

    def _check_supplier_vectors(
        self,
        rule: _Accumulator,
        actual_vectors: Mapping[
            tuple[Any, Any, Any], Mapping[tuple[Any, Any, Any], float]
        ],
    ) -> None:
        intent = self.intent
        if intent is None or not intent.intended_suppliers:
            rule.applicability = "not_applicable"
            return
        rule.tolerance = intent.tolerance
        for target_key, expected_entries in intent.intended_suppliers.items():
            rule.checked += 1
            expected = {tuple(key): float(amount) for key, amount in expected_entries}
            actual = dict(actual_vectors.get(tuple(target_key), {}))
            all_keys = set(expected) | set(actual)
            wrong = {
                key: {
                    "expected": expected.get(key, 0.0),
                    "actual": actual.get(key, 0.0),
                }
                for key in all_keys
                if not math.isclose(
                    expected.get(key, 0.0),
                    actual.get(key, 0.0),
                    rel_tol=intent.tolerance,
                    abs_tol=intent.tolerance,
                )
            }
            if wrong:
                rule.issue(
                    "Supplier vector composition differs from the independently declared target.",
                    activity_key=tuple(target_key),
                    expected=expected,
                    actual={"vector": actual, "differences": wrong},
                    tolerance=intent.tolerance,
                )

    def _check_algorithm(self, rule: _Accumulator) -> None:
        intent = self.intent
        if intent is None or intent.algorithm is None:
            rule.applicability = "not_applicable"
            return
        rule.checked = 1
        algorithm = intent.algorithm.lower().replace("_", "-")
        if self.system_model == "consequential" and "marginal" not in algorithm:
            rule.issue(
                "Consequential electricity and fuel markets must use a marginal-mix algorithm.",
                expected="marginal mix",
                actual=intent.algorithm,
            )

    def _check_coverage(
        self,
        rule: _Accumulator,
        keys_by_id: Mapping[int, tuple[Any, Any, Any]],
    ) -> None:
        intent = self.intent
        if intent is None or not (
            intent.expected_regions or intent.expected_technologies
        ):
            rule.applicability = "not_applicable"
            return
        keys = tuple(keys_by_id.values())
        for region in intent.expected_regions:
            rule.checked += 1
            if not any(key[2] == region for key in keys):
                rule.issue(
                    "Expected region has no validation target.",
                    expected=region,
                    actual="missing",
                )
        for technology in intent.expected_technologies:
            rule.checked += 1
            if not any(technology.lower() in str(key[0]).lower() for key in keys):
                rule.issue(
                    "Expected technology has no validation target.",
                    expected=technology,
                    actual="missing",
                )

    def _apply_suppressions(self, result: ValidationRuleResult) -> ValidationRuleResult:
        issues = []
        for issue in result.issues:
            suppression = next(
                (
                    item
                    for item in self.suppressions
                    if item.applies(
                        issue,
                        version=self.version,
                        system_model=self.system_model,
                    )
                ),
                None,
            )
            if suppression is not None:
                issue = replace(
                    issue,
                    suppressed=True,
                    suppression_explanation=suppression.explanation,
                )
            issues.append(issue)
        return replace(result, issues=tuple(issues))


def validation_cache_key(
    *,
    store_generation: int,
    scenario_identity: Any,
    source_fingerprint: str,
    iam_fingerprint: str,
    system_model: str,
    version: str,
    intent: ValidationIntent | None = None,
    baseline_cycles: Iterable[Iterable[tuple[Any, Any, Any]]] = (),
) -> str:
    return _stable_hash(
        {
            "store_generation": int(store_generation),
            "scenario_identity": scenario_identity,
            "source_fingerprint": source_fingerprint,
            "iam_fingerprint": iam_fingerprint,
            "system_model": system_model,
            "version": version,
            "ruleset_version": VALIDATION_RULESET_VERSION,
            "intent": intent,
            "baseline_cycles": baseline_cycles,
        }
    )


__all__ = [
    "VALIDATION_RULESET_VERSION",
    "ActivitySelector",
    "InventoryGraphValidator",
    "PremiseValidationError",
    "ValidationCertificate",
    "ValidationIntent",
    "ValidationIssue",
    "ValidationReport",
    "ValidationRuleResult",
    "ValidationSuppression",
    "inventory_activity_fingerprints",
    "inventory_cycle_signatures",
    "inventory_store_fingerprint",
    "load_validation_suppressions",
    "validation_cache_key",
]
