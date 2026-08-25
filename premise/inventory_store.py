"""Inventory storage abstractions used by :class:`premise.NewDatabase`.

The store is deliberately stricter than the historical ``list[dict]`` API:
readers receive immutable snapshots and writers must use a transaction.  The
legacy implementation is the semantic oracle; the compact implementation adds
copy-on-write forks and a versioned, columnar checkpoint representation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, TypeAlias

import numpy as np

try:  # PyArrow is a premise dependency, but keep source-only installs usable.
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except ImportError:  # pragma: no cover - exercised only in minimal environments
    pa = None
    pa_ipc = None


ActivityId: TypeAlias = int
ExchangeId: TypeAlias = int
STORE_SCHEMA_VERSION = 4
_UNHASHABLE = object()

_ACTIVITY_COMMON_FIELDS = (
    "name",
    "reference product",
    "product",
    "location",
    "unit",
    "database",
    "code",
    "type",
)
_EXCHANGE_STRING_FIELDS = (
    "name",
    "product",
    "location",
    "unit",
    "type",
)
_EXCHANGE_NUMERIC_FIELDS = ("amount",)
_EXCHANGE_BOOLEAN_FIELDS = ()
_NUMERIC_MISSING = 0
_NUMERIC_PYTHON_FLOAT = 1
_NUMERIC_PYTHON_INT = 2
_NUMERIC_FLOAT32 = 3
_NUMERIC_FLOAT64 = 4
_CandidatePositions: TypeAlias = set[int] | frozenset[int]


def _numeric_column_parts(value: Any) -> tuple[int, float | None, int | None]:
    if type(value) is float:
        return _NUMERIC_PYTHON_FLOAT, value, None
    if type(value) is int and -(2**63) <= value < 2**63:
        return _NUMERIC_PYTHON_INT, None, value
    if type(value) is np.float32:
        return _NUMERIC_FLOAT32, float(value), None
    if type(value) is np.float64:
        return _NUMERIC_FLOAT64, float(value), None
    return _NUMERIC_MISSING, None, None


def _decode_numeric_column(kind: int, float_value: Any, int_value: Any) -> Any:
    if kind == _NUMERIC_PYTHON_FLOAT:
        return float(float_value)
    if kind == _NUMERIC_PYTHON_INT:
        return int(int_value)
    if kind == _NUMERIC_FLOAT32:
        return np.float32(float_value)
    if kind == _NUMERIC_FLOAT64:
        return np.float64(float_value)
    raise ValueError(f"Unknown numeric exchange value kind: {kind}")


def _exchange_sidecar_metadata(
    payload: Mapping[str, Any],
    *,
    numeric_kinds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return only fields which cannot be restored losslessly from Arrow."""

    metadata = {}
    for key, value in payload.items():
        if key in _EXCHANGE_STRING_FIELDS and isinstance(value, str):
            continue
        if key in _EXCHANGE_NUMERIC_FIELDS:
            kind = (
                numeric_kinds[key]
                if numeric_kinds is not None and key in numeric_kinds
                else _numeric_column_parts(value)[0]
            )
            if kind != _NUMERIC_MISSING:
                continue
        if (
            key == "categories"
            and isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(item, str) for item in value)
        ):
            continue
        if key in _EXCHANGE_BOOLEAN_FIELDS and type(value) is bool:
            continue
        metadata[key] = value
    return metadata


class InventoryStoreError(RuntimeError):
    """Base class for inventory-store failures."""


class InventoryStoreCorruptionError(InventoryStoreError):
    """Raised when a checkpoint checksum or payload is invalid."""


class InventoryStoreVersionError(InventoryStoreError):
    """Raised when a checkpoint uses an unsupported schema."""


class InventoryStoreReadOnlyError(InventoryStoreError):
    """Raised when mutation is requested through a read-only view."""


@dataclass(frozen=True, slots=True)
class _CompiledWurstFilter:
    operation: str
    field_name: str | None = None
    value: Any = None
    children: tuple["_CompiledWurstFilter", ...] = ()

    def __call__(self, payload: Mapping[str, Any]) -> bool:
        if self.operation == "equals":
            return payload.get(self.field_name) == self.value
        if self.operation == "contains":
            return self.value in payload.get(self.field_name)
        if self.operation == "startswith":
            return payload.get(self.field_name, "").startswith(self.value)
        if self.operation == "either":
            return any(child(payload) for child in self.children)
        if self.operation == "exclude":
            return not self.children[0](payload)
        if self.operation == "doesnt-contain-any":
            return all(
                value not in payload.get(self.field_name) for value in self.value
            )
        raise ValueError(f"Unknown compiled Wurst operation: {self.operation}")


class IndexedInventoryList(list):
    """List-compatible inventory with lazy, order-preserving query indexes.

    It is a migration bridge for transformation code which still calls Wurst.
    Structural list changes invalidate the indexes. Sector wrappers re-wrap the
    result at every boundary, which also captures direct changes to indexed
    dataset fields.
    """

    _INDEXED_FIELDS = (
        "name",
        "reference product",
        "product",
        "location",
        "unit",
        "type",
        "database",
        "code",
    )

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._query_indexes = None
        self._indexed_query_fields: set[str] = set()

    def _invalidate(self) -> None:
        self._query_indexes = None
        self._indexed_query_fields = set()

    def _index_item(self, position: int, dataset: Mapping[str, Any]) -> None:
        if self._query_indexes is None:
            return
        exact, strings, predicate_cache = self._query_indexes
        predicate_cache.clear()
        for field_name in self._indexed_query_fields:
            value = dataset.get(field_name)
            hashable = _hashable(value)
            if hashable is not _UNHASHABLE:
                exact[field_name][hashable].add(position)
            if isinstance(value, str):
                strings[field_name][value].add(position)

    def _indexes(self, field_names: Iterable[str] = ()):
        if self._query_indexes is None:
            self._query_indexes = {}, {}, {}

        exact, strings, predicate_cache = self._query_indexes
        missing_fields = [
            field_name
            for field_name in field_names
            if field_name in self._INDEXED_FIELDS
            and field_name not in self._indexed_query_fields
        ]
        for field_name in missing_fields:
            field_exact: dict[Any, set[int]] = defaultdict(set)
            field_strings: dict[str, set[int]] = defaultdict(set)
            for position, dataset in enumerate(self):
                value = dataset.get(field_name)
                hashable = _hashable(value)
                if hashable is not _UNHASHABLE:
                    field_exact[hashable].add(position)
                if isinstance(value, str):
                    field_strings[value].add(position)
            exact[field_name] = field_exact
            strings[field_name] = field_strings
            self._indexed_query_fields.add(field_name)
        return self._query_indexes

    def _all_positions(self) -> frozenset[int]:
        _, _, predicate_cache = self._indexes()
        cache_key = ("all-positions", len(self))
        cached = predicate_cache.get(cache_key)
        if cached is None:
            cached = frozenset(range(len(self)))
            predicate_cache[cache_key] = cached
        return cached

    def _candidate_positions(
        self, expression: _CompiledWurstFilter
    ) -> _CandidatePositions | None:
        operation = expression.operation
        field_name = expression.field_name
        if operation == "equals" and field_name in self._INDEXED_FIELDS:
            exact, _, _ = self._indexes((field_name,))
            value = _hashable(expression.value)
            if value is _UNHASHABLE:
                return None
            # Query callers treat candidates as read-only. Returning the index
            # set avoids copying it for every exact predicate.
            return exact[field_name].get(value, frozenset())
        if (
            operation in {"contains", "startswith"}
            and field_name in self._INDEXED_FIELDS
        ):
            _, strings, predicate_cache = self._indexes((field_name,))
            cache_key = (operation, field_name, expression.value)
            cached = predicate_cache.get(cache_key)
            if cached is not None:
                return cached
            result: set[int] = set()
            for value, positions in strings[field_name].items():
                matches = (
                    expression.value in value
                    if operation == "contains"
                    else value.startswith(expression.value)
                )
                if matches:
                    result.update(positions)
            candidates = frozenset(result)
            predicate_cache[cache_key] = candidates
            return candidates
        if operation == "either":
            _, _, predicate_cache = self._indexes()
            cache_key = ("compiled", expression)
            try:
                cached = predicate_cache.get(cache_key)
            except TypeError:
                cache_key = None
                cached = None
            if cached is not None:
                return cached
            candidates = [
                self._candidate_positions(child) for child in expression.children
            ]
            if any(candidate is None for candidate in candidates):
                return None
            result: set[int] = set()
            for candidate in candidates:
                result.update(candidate)
            immutable_result = frozenset(result)
            if cache_key is not None:
                predicate_cache[cache_key] = immutable_result
            return immutable_result
        if operation == "exclude":
            _, _, predicate_cache = self._indexes()
            cache_key = ("compiled", expression)
            try:
                cached = predicate_cache.get(cache_key)
            except TypeError:
                cache_key = None
                cached = None
            if cached is not None:
                return cached
            candidates = self._candidate_positions(expression.children[0])
            if candidates is None:
                return None
            result = self._all_positions().difference(candidates)
            if cache_key is not None:
                predicate_cache[cache_key] = result
            return result
        if operation == "doesnt-contain-any" and field_name in self._INDEXED_FIELDS:
            _, strings, predicate_cache = self._indexes((field_name,))
            cache_key = (operation, field_name, expression.value)
            cached = predicate_cache.get(cache_key)
            if cached is not None:
                return cached
            excluded: set[int] = set()
            for value, positions in strings[field_name].items():
                if any(item in value for item in expression.value):
                    excluded.update(positions)
            result = self._all_positions().difference(excluded)
            predicate_cache[cache_key] = frozenset(result)
            return result
        return None

    def query_wurst(self, filters: tuple[_CompiledWurstFilter, ...]):
        candidate_groups: list[_CandidatePositions] = []
        for expression in filters:
            positions = self._candidate_positions(expression)
            if positions is None:
                return None
            candidate_groups.append(positions)
        if not candidate_groups:
            candidates: Iterable[int] = range(len(self))
        elif len(candidate_groups) == 1:
            candidates = candidate_groups[0]
        else:
            candidate_groups.sort(key=len)
            intersection = set(candidate_groups[0])
            for positions in candidate_groups[1:]:
                intersection.intersection_update(positions)
                if not intersection:
                    break
            candidates = intersection
        return (
            self[position]
            for position in sorted(candidates)
            if all(expression(self[position]) for expression in filters)
        )

    def append(self, item) -> None:
        super().append(item)
        self._index_item(len(self) - 1, item)

    def extend(self, iterable) -> None:
        additions = list(iterable)
        start = len(self)
        super().extend(additions)
        for offset, item in enumerate(additions):
            self._index_item(start + offset, item)

    def insert(self, index, item) -> None:
        super().insert(index, item)
        self._invalidate()

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._invalidate()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._invalidate()

    def __iadd__(self, iterable):
        self.extend(iterable)
        return self

    def pop(self, index=-1):
        value = super().pop(index)
        self._invalidate()
        return value

    def remove(self, value) -> None:
        super().remove(value)
        self._invalidate()

    def clear(self) -> None:
        super().clear()
        self._invalidate()

    def reverse(self) -> None:
        super().reverse()
        self._invalidate()

    def sort(self, *args, **kwargs) -> None:
        super().sort(*args, **kwargs)
        self._invalidate()


def install_wurst_query_engine() -> bool:
    """Install metadata-carrying Wurst predicates and an indexed ``get_many``.

    Plain lists and dynamic predicates retain Wurst's original ordered scan.
    The patch is idempotent and affects no query results.
    """

    try:
        from wurst import searching as ws
    except ImportError:  # pragma: no cover - premise normally depends on Wurst
        return False
    if getattr(ws, "_premise_inventory_store_query_engine", False):
        return True

    original = {
        "equals": ws.equals,
        "contains": ws.contains,
        "startswith": ws.startswith,
        "either": ws.either,
        "exclude": ws.exclude,
        "doesnt_contain_any": ws.doesnt_contain_any,
        "get_many": ws.get_many,
    }

    def equals(field_name, value):
        return _CompiledWurstFilter("equals", field_name, value)

    def contains(field_name, value):
        return _CompiledWurstFilter("contains", field_name, value)

    def startswith(field_name, value):
        return _CompiledWurstFilter("startswith", field_name, value)

    def either(*filters):
        if all(isinstance(item, _CompiledWurstFilter) for item in filters):
            return _CompiledWurstFilter("either", children=tuple(filters))
        return original["either"](*filters)

    def exclude(filter_expression):
        if isinstance(filter_expression, _CompiledWurstFilter):
            return _CompiledWurstFilter("exclude", children=(filter_expression,))
        return original["exclude"](filter_expression)

    def doesnt_contain_any(field_name, values):
        return _CompiledWurstFilter("doesnt-contain-any", field_name, tuple(values))

    def get_many(data, *filters):
        if isinstance(data, IndexedInventoryList):
            if all(isinstance(item, _CompiledWurstFilter) for item in filters):
                result = data.query_wurst(tuple(filters))
                if result is not None:
                    ws._premise_inventory_store_query_diagnostics["indexed"] += 1
                    return result
            ws._premise_inventory_store_query_diagnostics["fallback"] += 1
        return original["get_many"](data, *filters)

    ws.equals = equals
    ws.contains = contains
    ws.startswith = startswith
    ws.either = either
    ws.exclude = exclude
    ws.doesnt_contain_any = doesnt_contain_any
    ws.get_many = get_many
    ws._premise_inventory_store_query_engine = True
    ws._premise_inventory_store_originals = original
    ws._premise_inventory_store_query_diagnostics = {
        "indexed": 0,
        "fallback": 0,
    }
    return True


install_wurst_query_engine()


def get_wurst_query_diagnostics(*, reset: bool = False) -> dict[str, int]:
    """Return counts of indexed and ordered-fallback Wurst scans."""

    from wurst import searching as ws

    diagnostics = dict(
        getattr(
            ws,
            "_premise_inventory_store_query_diagnostics",
            {"indexed": 0, "fallback": 0},
        )
    )
    if reset:
        ws._premise_inventory_store_query_diagnostics = {
            "indexed": 0,
            "fallback": 0,
        }
    return diagnostics


@dataclass(frozen=True, slots=True)
class ActivityKey:
    name: str
    product: str
    location: str


@dataclass(frozen=True, slots=True)
class ProviderKey:
    name: str
    product: str
    unit: str


@dataclass(frozen=True, slots=True)
class FilterExpression:
    """A serialisable activity predicate.

    ``operator`` accepts ``equals``/``exact``, ``contains``, ``startswith``,
    ``in``/``either``, ``all``, and their common negative variants.  A callable
    value is supported explicitly as an ordered fallback scan.
    """

    field: str
    value: Any
    operator: str = "equals"


@dataclass(frozen=True, slots=True)
class ActivityQuery:
    filters: tuple[FilterExpression, ...]
    masks: tuple[FilterExpression, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", tuple(self.filters))
        object.__setattr__(self, "masks", tuple(self.masks))


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


class _ImmutableRecord(Mapping[str, Any]):
    __slots__ = ("_data", "_raw")

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._raw = copy.deepcopy(dict(data))
        self._data = _freeze(self._raw)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self._data)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return an independent mutable copy of this snapshot."""

        return copy.deepcopy(self._raw)


class ExchangeRecord(_ImmutableRecord):
    __slots__ = ("exchange_id", "activity_id")

    def __init__(
        self,
        exchange_id: ExchangeId,
        activity_id: ActivityId,
        data: Mapping[str, Any],
    ) -> None:
        super().__init__(data)
        self.exchange_id = exchange_id
        self.activity_id = activity_id

    @property
    def id(self) -> ExchangeId:
        return self.exchange_id


class ActivityRecord(_ImmutableRecord):
    __slots__ = ("activity_id", "exchange_ids")

    def __init__(
        self,
        activity_id: ActivityId,
        data: Mapping[str, Any],
        exchange_ids: Iterable[ExchangeId] = (),
    ) -> None:
        super().__init__(data)
        self.activity_id = activity_id
        self.exchange_ids = tuple(exchange_ids)

    @property
    def id(self) -> ActivityId:
        return self.activity_id


@dataclass
class _StoreState:
    activities: dict[ActivityId, dict[str, Any]] = field(default_factory=dict)
    activity_order: list[ActivityId] = field(default_factory=list)
    exchanges: dict[ExchangeId, dict[str, Any]] = field(default_factory=dict)
    exchange_owner: dict[ExchangeId, ActivityId] = field(default_factory=dict)
    activity_exchanges: dict[ActivityId, list[ExchangeId] | range] = field(
        default_factory=dict
    )
    next_activity_id: int = 0
    next_exchange_id: int = 0
    generation: int = 0
    transaction_log: list[str] = field(default_factory=list)
    field_index: dict[str, dict[Any, tuple[ActivityId, ...]]] = field(
        default_factory=dict
    )
    activity_key_index: dict[ActivityKey, tuple[ActivityId, ...]] = field(
        default_factory=dict
    )
    provider_index: dict[ProviderKey, tuple[ActivityId, ...]] = field(
        default_factory=dict
    )
    consumer_index: dict[ActivityId, tuple[ActivityId, ...]] = field(
        default_factory=dict
    )
    indexes_ready: bool = False


class _DenseExchangeTable:
    """Dense integer-ID exchange table with stable tombstoned positions."""

    __slots__ = ("_rows", "_length")

    def __init__(self) -> None:
        self._rows: list[dict[str, Any] | None] = []
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def __contains__(self, exchange_id: object) -> bool:
        return (
            isinstance(exchange_id, int)
            and 0 <= exchange_id < len(self._rows)
            and self._rows[exchange_id] is not None
        )

    def __getitem__(self, exchange_id: ExchangeId) -> dict[str, Any]:
        if exchange_id not in self:
            raise KeyError(exchange_id)
        return self._rows[exchange_id]

    def __setitem__(self, exchange_id: ExchangeId, payload: dict[str, Any]) -> None:
        if exchange_id == len(self._rows):
            self._rows.append(payload)
            self._length += 1
            return
        if not 0 <= exchange_id < len(self._rows):
            raise IndexError(f"Non-contiguous exchange id: {exchange_id}")
        if self._rows[exchange_id] is None:
            self._length += 1
        self._rows[exchange_id] = payload

    def __delitem__(self, exchange_id: ExchangeId) -> None:
        if exchange_id not in self:
            raise KeyError(exchange_id)
        self._rows[exchange_id] = None
        self._length -= 1

    def shallow_copy(self) -> "_DenseExchangeTable":
        """Copy the row index while sharing immutable transaction payloads."""

        duplicate = type(self)()
        duplicate._rows = self._rows.copy()
        duplicate._length = self._length
        return duplicate


def _product(payload: Mapping[str, Any]) -> Any:
    return payload.get("reference product", payload.get("product"))


def _hashable(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return _UNHASHABLE
    return value


def _normalise_query(
    query: ActivityQuery | FilterExpression | Mapping[str, Any] | Callable | None,
) -> ActivityQuery | Callable | None:
    if query is None or isinstance(query, ActivityQuery) or callable(query):
        return query
    if isinstance(query, FilterExpression):
        return ActivityQuery((query,))
    if isinstance(query, Mapping):
        return ActivityQuery(
            tuple(
                FilterExpression(field_name, value)
                for field_name, value in query.items()
            )
        )
    raise TypeError(
        "query must be an ActivityQuery, FilterExpression, mapping, callable, or None"
    )


def _matches_expression(
    payload: Mapping[str, Any], expression: FilterExpression
) -> bool:
    actual = payload.get(expression.field)
    expected = expression.value
    operator = expression.operator.lower().replace("_", "-")

    if callable(expected):
        result = bool(expected(actual))
    elif operator in {"equals", "equal", "exact", "=="}:
        result = actual == expected
    elif operator in {"contains", "contain"}:
        if actual is None:
            result = False
        elif isinstance(expected, (tuple, list, set, frozenset)):
            result = any(item in actual for item in expected)
        else:
            result = expected in actual
    elif operator in {"startswith", "starts-with", "prefix"}:
        result = isinstance(actual, str) and actual.startswith(expected)
    elif operator in {"in", "either", "one-of"}:
        result = actual in expected
    elif operator == "all":
        result = actual is not None and all(item in actual for item in expected)
    elif operator in {"not-equals", "not-equal", "!=", "exclusion", "exclude"}:
        result = actual != expected
    elif operator in {"not-contains", "doesnt-contain", "does-not-contain"}:
        result = actual is None or expected not in actual
    elif operator in {"not-in", "neither"}:
        result = actual not in expected
    else:
        raise ValueError(f"Unsupported filter operator: {expression.operator!r}")
    return result


class InventoryStore(ABC):
    """Abstract ordered inventory graph."""

    backend_name: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def iter_activity_ids(self) -> Iterator[ActivityId]: ...

    @abstractmethod
    def iter_activities(
        self,
        query: (
            ActivityQuery | FilterExpression | Mapping[str, Any] | Callable | None
        ) = None,
    ) -> Iterator[ActivityRecord]: ...

    @abstractmethod
    def activity(self, activity_id: ActivityId) -> ActivityRecord: ...

    @abstractmethod
    def exchange(self, exchange_id: ExchangeId) -> ExchangeRecord: ...

    @abstractmethod
    def find(
        self, query: ActivityQuery | FilterExpression | Mapping[str, Any] | Callable
    ) -> tuple[ActivityRecord, ...]: ...

    @abstractmethod
    def find_one(
        self, query: ActivityQuery | FilterExpression | Mapping[str, Any] | Callable
    ) -> ActivityRecord: ...

    @abstractmethod
    def providers(
        self, provider_key: ProviderKey, consumer_location: str | None = None
    ) -> tuple[ActivityRecord, ...]: ...

    @abstractmethod
    def consumers(self, activity_id: ActivityId) -> tuple[ActivityId, ...]: ...

    @abstractmethod
    def contains(self, activity_key: ActivityKey) -> bool: ...

    @abstractmethod
    def transaction(self, label: str) -> "InventoryTransaction": ...

    @abstractmethod
    def fork(self, scenario_identity: Any = None) -> "InventoryStore": ...

    @abstractmethod
    def checkpoint(self, path: str | Path) -> Path: ...

    @abstractmethod
    def iter_materialized(
        self, restore_metadata: bool = True
    ) -> Iterator[dict[str, Any]]: ...

    def materialize(self, restore_metadata: bool = True) -> list[dict[str, Any]]:
        """Materialise the complete graph.

        This is intentionally explicit: for a production ecoinvent graph the
        returned Python dictionaries can require several gigabytes of memory.
        """

        return list(self.iter_materialized(restore_metadata=restore_metadata))

    @classmethod
    def open(cls, path: str | Path) -> "InventoryStore":
        """Open and validate a versioned inventory checkpoint."""

        return _open_checkpoint(Path(path))


class _InMemoryInventoryStore(InventoryStore):
    backend_name = "legacy"
    eager_indexes = True
    eager_exchange_owners = True
    dense_exchange_table = False

    def __init__(
        self,
        database: Iterable[Mapping[str, Any]] = (),
        *,
        scenario_identity: Any = None,
        take_ownership: bool = False,
    ) -> None:
        self._state = self._new_state()
        self._scenario_identity = scenario_identity
        self._lock = threading.RLock()
        self._active_transaction = False
        self._shared_state = False
        self._ingest(database, take_ownership=take_ownership)

    def _new_state(self) -> _StoreState:
        state = _StoreState()
        if self.dense_exchange_table:
            state.exchanges = _DenseExchangeTable()
        return state

    @property
    def generation(self) -> int:
        return self._state.generation

    @property
    def scenario_identity(self) -> Any:
        return self._scenario_identity

    def _ingest(
        self,
        database: Iterable[Mapping[str, Any]],
        *,
        take_ownership: bool,
    ) -> None:
        for dataset in database:
            payload = (
                dataset
                if take_ownership and isinstance(dataset, dict)
                else copy.deepcopy(dict(dataset))
            )
            exchanges = payload.pop("exchanges", [])
            activity_id = self._state.next_activity_id
            self._state.next_activity_id += 1
            self._state.activities[activity_id] = payload
            self._state.activity_order.append(activity_id)
            if self.dense_exchange_table:
                exchange_start = self._state.next_exchange_id
                for exchange in exchanges:
                    exchange_id = self._state.next_exchange_id
                    self._state.next_exchange_id += 1
                    self._state.exchanges[exchange_id] = (
                        exchange
                        if take_ownership and isinstance(exchange, dict)
                        else copy.deepcopy(dict(exchange))
                    )
                self._state.activity_exchanges[activity_id] = range(
                    exchange_start, self._state.next_exchange_id
                )
            else:
                self._state.activity_exchanges[activity_id] = []
                for exchange in exchanges:
                    self._add_exchange_unchecked(
                        activity_id, exchange, take_ownership=take_ownership
                    )
        if self.eager_indexes:
            self._rebuild_indexes()

    def _ensure_owned_state(self) -> None:
        if self._shared_state:
            self._state = (
                self._copy_compact_state()
                if self.backend_name == "compact"
                else copy.deepcopy(self._state)
            )
            self._shared_state = False

    def _copy_compact_state(self) -> _StoreState:
        """Copy graph structure while retaining immutable payload snapshots."""

        state = self._state
        duplicate = copy.copy(state)
        duplicate.activities = state.activities.copy()
        duplicate.activity_order = state.activity_order.copy()
        duplicate.exchanges = state.exchanges.shallow_copy()
        duplicate.exchange_owner = state.exchange_owner.copy()
        duplicate.activity_exchanges = state.activity_exchanges.copy()
        duplicate.transaction_log = state.transaction_log.copy()
        return duplicate

    def _transaction_snapshot(self) -> _StoreState:
        """Return a rollback snapshot without copying every compact payload.

        Transaction mutation methods replace activity and exchange dictionaries
        instead of changing existing payloads in place. A shallow structural
        snapshot is therefore sufficient for compact stores and keeps a small
        sector transaction from duplicating the complete inventory graph.
        Legacy retains its historical deep-copy oracle semantics.
        """

        if self.backend_name != "compact":
            return copy.deepcopy(self._state)

        return self._copy_compact_state()

    def _invalidate_indexes(self) -> None:
        self._state.field_index = {}
        self._state.activity_key_index = {}
        self._state.provider_index = {}
        self._state.consumer_index = {}
        self._state.indexes_ready = False

    def _ensure_indexes(self) -> None:
        if self._state.indexes_ready:
            return
        with self._lock:
            if not self._state.indexes_ready:
                self._rebuild_indexes()

    def _ensure_exchange_owners(self) -> None:
        if len(self._state.exchange_owner) == len(self._state.exchanges):
            return
        with self._lock:
            if len(self._state.exchange_owner) == len(self._state.exchanges):
                return
            self._state.exchange_owner = {
                exchange_id: activity_id
                for activity_id in self.iter_activity_ids()
                for exchange_id in self._state.activity_exchanges[activity_id]
                if exchange_id in self._state.exchanges
            }

    def __len__(self) -> int:
        return len(self._state.activities)

    def iter_activity_ids(self) -> Iterator[ActivityId]:
        for activity_id in self._state.activity_order:
            if activity_id in self._state.activities:
                yield activity_id

    def _activity_payload(self, activity_id: ActivityId) -> dict[str, Any]:
        try:
            payload = self._state.activities[activity_id]
        except KeyError as error:
            raise KeyError(f"Unknown activity id: {activity_id}") from error
        record = copy.deepcopy(payload)
        record["exchanges"] = [
            copy.deepcopy(self._state.exchanges[exchange_id])
            for exchange_id in self._state.activity_exchanges[activity_id]
            if exchange_id in self._state.exchanges
        ]
        return record

    def _iter_storage_activities(
        self,
    ) -> Iterator[tuple[ActivityId, Mapping[str, Any], tuple[ExchangeId, ...]]]:
        """Yield read-only activity metadata for package-native kernels.

        This private interface avoids materialising exchange dictionaries when
        a migrated sector only needs activity metadata and stable exchange IDs.
        Mutations still have to go through :class:`InventoryTransaction`.
        """

        for activity_id in self.iter_activity_ids():
            yield (
                activity_id,
                MappingProxyType(self._state.activities[activity_id]),
                tuple(self._state.activity_exchanges[activity_id]),
            )

    def _storage_exchange(self, exchange_id: ExchangeId) -> Mapping[str, Any]:
        """Return a read-only exchange mapping for a package-native kernel."""

        if exchange_id not in self._state.exchanges:
            raise KeyError(f"Unknown exchange id: {exchange_id}")
        return MappingProxyType(self._state.exchanges[exchange_id])

    def activity(self, activity_id: ActivityId) -> ActivityRecord:
        return ActivityRecord(
            activity_id,
            self._activity_payload(activity_id),
            self._state.activity_exchanges[activity_id],
        )

    def exchange(self, exchange_id: ExchangeId) -> ExchangeRecord:
        self._ensure_exchange_owners()
        try:
            payload = self._state.exchanges[exchange_id]
            owner = self._state.exchange_owner[exchange_id]
        except KeyError as error:
            raise KeyError(f"Unknown exchange id: {exchange_id}") from error
        return ExchangeRecord(exchange_id, owner, copy.deepcopy(payload))

    def _candidate_ids(self, query: ActivityQuery) -> Iterable[ActivityId]:
        self._ensure_indexes()
        candidate: set[ActivityId] | None = None
        for expression in query.filters:
            if expression.operator.lower() not in {"equals", "equal", "exact", "=="}:
                continue
            value = _hashable(expression.value)
            if value is _UNHASHABLE:
                continue
            indexed = set(
                self._state.field_index.get(expression.field, {}).get(value, ())
            )
            candidate = (
                indexed if candidate is None else candidate.intersection(indexed)
            )
        if candidate is None:
            return self.iter_activity_ids()
        return (
            activity_id
            for activity_id in self.iter_activity_ids()
            if activity_id in candidate
        )

    def iter_activities(
        self,
        query: (
            ActivityQuery | FilterExpression | Mapping[str, Any] | Callable | None
        ) = None,
    ) -> Iterator[ActivityRecord]:
        normalised = _normalise_query(query)
        if callable(normalised):
            for activity_id in self.iter_activity_ids():
                record = self.activity(activity_id)
                if normalised(record):
                    yield record
            return

        if normalised is None:
            for activity_id in self.iter_activity_ids():
                yield self.activity(activity_id)
            return

        for activity_id in self._candidate_ids(normalised):
            payload = self._state.activities[activity_id]
            if not all(
                _matches_expression(payload, expression)
                for expression in normalised.filters
            ):
                continue
            if any(
                _matches_expression(payload, expression)
                for expression in normalised.masks
            ):
                continue
            yield self.activity(activity_id)

    def find(
        self,
        query: ActivityQuery | FilterExpression | Mapping[str, Any] | Callable,
    ) -> tuple[ActivityRecord, ...]:
        return tuple(self.iter_activities(query))

    def find_one(
        self,
        query: ActivityQuery | FilterExpression | Mapping[str, Any] | Callable,
    ) -> ActivityRecord:
        records = self.find(query)
        if len(records) != 1:
            raise ValueError(f"Expected exactly one activity, found {len(records)}.")
        return records[0]

    def providers(
        self, provider_key: ProviderKey, consumer_location: str | None = None
    ) -> tuple[ActivityRecord, ...]:
        self._ensure_indexes()
        activity_ids = list(self._state.provider_index.get(provider_key, ()))
        if consumer_location is not None:
            location_rank = {
                consumer_location: 0,
                "RER": 1,
                "RoW": 2,
                "GLO": 3,
            }
            activity_ids.sort(
                key=lambda activity_id: (
                    location_rank.get(
                        self._state.activities[activity_id].get("location"), 4
                    ),
                    self._state.activity_order.index(activity_id),
                )
            )
        return tuple(self.activity(activity_id) for activity_id in activity_ids)

    def consumers(self, activity_id: ActivityId) -> tuple[ActivityId, ...]:
        if activity_id not in self._state.activities:
            raise KeyError(f"Unknown activity id: {activity_id}")
        self._ensure_indexes()
        return self._state.consumer_index.get(activity_id, ())

    def contains(self, activity_key: ActivityKey) -> bool:
        self._ensure_indexes()
        return bool(self._state.activity_key_index.get(activity_key))

    def transaction(self, label: str) -> "InventoryTransaction":
        return InventoryTransaction(self, label)

    def fork(self, scenario_identity: Any = None) -> "InventoryStore":
        child = object.__new__(type(self))
        child._state = copy.deepcopy(self._state)
        child._scenario_identity = scenario_identity
        child._lock = threading.RLock()
        child._active_transaction = False
        child._shared_state = False
        return child

    def iter_materialized(
        self, restore_metadata: bool = True
    ) -> Iterator[dict[str, Any]]:
        del restore_metadata  # All metadata is retained losslessly in this backend.
        for activity_id in self.iter_activity_ids():
            yield self._activity_payload(activity_id)

    def checkpoint(self, path: str | Path) -> Path:
        return _write_checkpoint(self, Path(path))

    def _add_activity_unchecked(self, payload: Mapping[str, Any]) -> ActivityId:
        data = copy.deepcopy(dict(payload))
        exchanges = data.pop("exchanges", [])
        activity_id = self._state.next_activity_id
        self._state.next_activity_id += 1
        self._state.activities[activity_id] = data
        self._state.activity_order.append(activity_id)
        self._state.activity_exchanges[activity_id] = []
        for exchange in exchanges:
            self._add_exchange_unchecked(activity_id, exchange)
        return activity_id

    def _add_exchange_unchecked(
        self,
        activity_id: ActivityId,
        payload: Mapping[str, Any],
        *,
        take_ownership: bool = False,
    ) -> ExchangeId:
        if activity_id not in self._state.activities:
            raise KeyError(f"Unknown activity id: {activity_id}")
        exchange_id = self._state.next_exchange_id
        self._state.next_exchange_id += 1
        self._state.exchanges[exchange_id] = (
            payload
            if take_ownership and isinstance(payload, dict)
            else copy.deepcopy(dict(payload))
        )
        if self.eager_exchange_owners:
            self._state.exchange_owner[exchange_id] = activity_id
        exchange_ids = self._state.activity_exchanges[activity_id]
        exchange_ids = list(exchange_ids)
        exchange_ids.append(exchange_id)
        self._state.activity_exchanges[activity_id] = exchange_ids
        return exchange_id

    def _remove_exchange_unchecked(self, exchange_id: ExchangeId) -> None:
        if exchange_id not in self._state.exchanges:
            raise KeyError(f"Unknown exchange id: {exchange_id}")
        self._ensure_exchange_owners()
        owner = self._state.exchange_owner.pop(exchange_id)
        exchange_ids = list(self._state.activity_exchanges[owner])
        exchange_ids.remove(exchange_id)
        self._state.activity_exchanges[owner] = exchange_ids
        del self._state.exchanges[exchange_id]

    def _rebuild_indexes(self) -> None:
        fields: dict[str, dict[Any, list[ActivityId]]] = defaultdict(
            lambda: defaultdict(list)
        )
        activity_keys: dict[ActivityKey, list[ActivityId]] = defaultdict(list)
        provider_keys: dict[ProviderKey, list[ActivityId]] = defaultdict(list)
        identifiers: dict[tuple[Any, Any], ActivityId] = {}

        for activity_id in self.iter_activity_ids():
            payload = self._state.activities[activity_id]
            for field_name, value in payload.items():
                value = _hashable(value)
                if value is not _UNHASHABLE:
                    fields[field_name][value].append(activity_id)
            name = payload.get("name")
            product = _product(payload)
            location = payload.get("location")
            unit = payload.get("unit")
            if None not in (name, product, location):
                activity_keys[ActivityKey(name, product, location)].append(activity_id)
            if None not in (name, product, unit):
                provider_keys[ProviderKey(name, product, unit)].append(activity_id)
            if payload.get("database") is not None and payload.get("code") is not None:
                identifiers[(payload["database"], payload["code"])] = activity_id

        consumers: dict[ActivityId, list[ActivityId]] = defaultdict(list)
        for consumer_id in self.iter_activity_ids():
            for exchange_id in self._state.activity_exchanges[consumer_id]:
                exchange = self._state.exchanges[exchange_id]
                provider_id = None
                exchange_input = exchange.get("input")
                if (
                    isinstance(exchange_input, (tuple, list))
                    and len(exchange_input) == 2
                ):
                    provider_id = identifiers.get(tuple(exchange_input))
                if provider_id is None and exchange.get("type") == "technosphere":
                    key = ActivityKey(
                        exchange.get("name"),
                        _product(exchange),
                        exchange.get("location"),
                    )
                    matches = activity_keys.get(key, ())
                    if len(matches) == 1:
                        provider_id = matches[0]
                if (
                    provider_id is not None
                    and consumer_id not in consumers[provider_id]
                ):
                    consumers[provider_id].append(consumer_id)

        self._state.field_index = {
            field_name: {value: tuple(ids) for value, ids in values.items()}
            for field_name, values in fields.items()
        }
        self._state.activity_key_index = {
            key: tuple(ids) for key, ids in activity_keys.items()
        }
        self._state.provider_index = {
            key: tuple(ids) for key, ids in provider_keys.items()
        }
        self._state.consumer_index = {key: tuple(ids) for key, ids in consumers.items()}
        self._state.indexes_ready = True


class LegacyInventoryStore(_InMemoryInventoryStore):
    """Exact, dictionary-backed oracle implementation."""

    backend_name = "legacy"


class CompactInventoryStore(_InMemoryInventoryStore):
    """Copy-on-write inventory store with columnar checkpoints.

    The first implementation intentionally shares the battle-tested mutation
    engine with the legacy oracle.  Forks share state until the first write;
    checkpoints split activities and exchanges and dictionary-encode common
    string columns when PyArrow is available.
    """

    backend_name = "compact"
    eager_indexes = False
    eager_exchange_owners = False
    dense_exchange_table = True

    def _checkout_materialized(
        self, *, discard_shared_state: bool = False
    ) -> IndexedInventoryList:
        """Transfer graph ownership to the private transformation bridge.

        This is intentionally internal. It is safe only when the caller owns
        the final reference to the source graph (or explicitly discards all
        other references), because returned dictionaries are mutable.
        """

        with self._lock:
            if self._active_transaction:
                raise InventoryStoreError(
                    "Cannot check out a store during an active transaction."
                )
            if self._shared_state and not discard_shared_state:
                raise InventoryStoreError(
                    "Cannot check out shared compact state without discarding "
                    "every other reference."
                )
            state = self._state
            database = IndexedInventoryList()
            # Bypass IndexedInventoryList.append: no query index exists yet,
            # and building it incrementally would only add overhead here.
            append = list.append
            for activity_id in state.activity_order:
                if activity_id not in state.activities:
                    continue
                payload = state.activities[activity_id]
                payload["exchanges"] = [
                    state.exchanges[exchange_id]
                    for exchange_id in state.activity_exchanges[activity_id]
                    if exchange_id in state.exchanges
                ]
                append(database, payload)
            self._state = self._new_state()
            self._shared_state = False
            return database

    def fork(self, scenario_identity: Any = None) -> "CompactInventoryStore":
        child = object.__new__(type(self))
        child._state = self._state
        child._scenario_identity = scenario_identity
        child._lock = threading.RLock()
        child._active_transaction = False
        child._shared_state = True
        self._shared_state = True
        return child


_SCENARIO_MAPPING_MISSING = object()
_SCENARIO_MAPPING_RESIDENT_FIELDS = (
    "name",
    "reference product",
    "product",
    "location",
    "unit",
    "lhv",
)
_SCENARIO_MAPPING_RESIDENT_ATTRIBUTES = tuple(
    field_name.replace(" ", "_") for field_name in _SCENARIO_MAPPING_RESIDENT_FIELDS
)
_SCENARIO_MAPPING_RESIDENT_FIELD_BITS = {
    field_name: 1 << position
    for position, field_name in enumerate(_SCENARIO_MAPPING_RESIDENT_FIELDS)
}


class _CheckpointActivityResolver:
    """Load uncommon scenario-mapping metadata only when it is requested."""

    __slots__ = ("checkpoint", "_store", "_lock")

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self._store: InventoryStore | None = None
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Path]:
        return {"checkpoint": self.checkpoint}

    def __setstate__(self, state: Mapping[str, Path]) -> None:
        self.checkpoint = state["checkpoint"]
        self._store = None
        self._lock = threading.Lock()

    def _payload(self, activity_id: ActivityId) -> Mapping[str, Any]:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = InventoryStore.open(self.checkpoint)
        try:
            return self._store._state.activities[activity_id]
        except KeyError as error:
            raise KeyError(f"Unknown checkpoint activity id: {activity_id}") from error

    def value(self, activity_id: ActivityId, field_name: str) -> Any:
        payload = self._payload(activity_id)
        if field_name not in payload:
            raise KeyError(field_name)
        return copy.deepcopy(payload[field_name])

    def keys(self, activity_id: ActivityId) -> tuple[str, ...]:
        return tuple(self._payload(activity_id))

    def contains(self, activity_id: ActivityId, field_name: object) -> bool:
        return field_name in self._payload(activity_id)


class _ScenarioActivityReference(Mapping[str, Any]):
    """Small activity mapping backed by a versioned inventory checkpoint."""

    __slots__ = (
        "_resolver",
        "_activity_id",
        "_resident_mask",
        *_SCENARIO_MAPPING_RESIDENT_ATTRIBUTES,
    )

    def __init__(
        self,
        resolver: _CheckpointActivityResolver,
        activity_id: ActivityId,
        payload: Mapping[str, Any],
    ) -> None:
        self._resolver = resolver
        self._activity_id = activity_id
        resident_mask = 0
        for field_name in _SCENARIO_MAPPING_RESIDENT_FIELDS:
            if field_name in payload:
                resident_mask |= _SCENARIO_MAPPING_RESIDENT_FIELD_BITS[field_name]
                value = copy.deepcopy(payload[field_name])
            else:
                value = None
            setattr(
                self,
                field_name.replace(" ", "_"),
                value,
            )
        self._resident_mask = resident_mask

    def _resident_value(self, field_name: str) -> Any:
        field_bit = _SCENARIO_MAPPING_RESIDENT_FIELD_BITS.get(field_name)
        if field_bit is None or not self._resident_mask & field_bit:
            return _SCENARIO_MAPPING_MISSING
        return getattr(self, field_name.replace(" ", "_"))

    def __getitem__(self, field_name: str) -> Any:
        resident = self._resident_value(field_name)
        if resident is not _SCENARIO_MAPPING_MISSING:
            return copy.deepcopy(resident)
        return self._resolver.value(self._activity_id, field_name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._resolver.keys(self._activity_id))

    def __len__(self) -> int:
        return len(self._resolver.keys(self._activity_id))

    def __contains__(self, field_name: object) -> bool:
        if isinstance(field_name, str):
            resident = self._resident_value(field_name)
            if resident is not _SCENARIO_MAPPING_MISSING:
                return True
            if field_name in _SCENARIO_MAPPING_RESIDENT_FIELDS:
                return False
        return self._resolver.contains(self._activity_id, field_name)


def _compact_scenario_mapping(
    mapping: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    store: CompactInventoryStore,
    checkpoint: Path,
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Replace store-owned mapping payloads with shared lazy references."""

    activity_ids = {
        id(payload): activity_id
        for activity_id, payload in store._state.activities.items()
    }
    resolver = _CheckpointActivityResolver(checkpoint)
    references: dict[ActivityId, _ScenarioActivityReference] = {}
    compacted: dict[str, dict[str, list[Mapping[str, Any]]]] = {}

    for sector, sector_mapping in mapping.items():
        compacted[sector] = {}
        for variable, activities in sector_mapping.items():
            compacted_activities: list[Mapping[str, Any]] = []
            for activity in activities:
                activity_id = activity_ids.get(id(activity))
                if activity_id is None:
                    compacted_activities.append(activity)
                    continue
                reference = references.get(activity_id)
                if reference is None:
                    reference = _ScenarioActivityReference(
                        resolver, activity_id, activity
                    )
                    references[activity_id] = reference
                compacted_activities.append(reference)
            compacted[sector][variable] = compacted_activities

    return compacted


def _hydrate_scenario_mapping(
    mapping: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    activities_by_id: Mapping[ActivityId, dict[str, Any]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Rebind lazy mapping entries to an incremental update's working graph."""

    hydrated: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for sector, sector_mapping in mapping.items():
        hydrated[sector] = {}
        for variable, activities in sector_mapping.items():
            hydrated[sector][variable] = [
                (
                    activities_by_id.get(activity._activity_id, activity)
                    if isinstance(activity, _ScenarioActivityReference)
                    else activity
                )
                for activity in activities
            ]
    return hydrated


class ReadOnlyInventoryStore(InventoryStore):
    """Read-only facade returned by ``NewDatabase.get_inventory_store``."""

    def __init__(self, store: InventoryStore) -> None:
        self._store = store
        self.backend_name = store.backend_name

    def __len__(self) -> int:
        return len(self._store)

    def iter_activity_ids(self) -> Iterator[ActivityId]:
        return self._store.iter_activity_ids()

    def iter_activities(self, query=None) -> Iterator[ActivityRecord]:
        return self._store.iter_activities(query)

    def activity(self, activity_id: ActivityId) -> ActivityRecord:
        return self._store.activity(activity_id)

    def exchange(self, exchange_id: ExchangeId) -> ExchangeRecord:
        return self._store.exchange(exchange_id)

    def find(self, query) -> tuple[ActivityRecord, ...]:
        return self._store.find(query)

    def find_one(self, query) -> ActivityRecord:
        return self._store.find_one(query)

    def providers(self, provider_key, consumer_location=None):
        return self._store.providers(provider_key, consumer_location)

    def consumers(self, activity_id: ActivityId) -> tuple[ActivityId, ...]:
        return self._store.consumers(activity_id)

    def contains(self, activity_key: ActivityKey) -> bool:
        return self._store.contains(activity_key)

    def transaction(self, label: str) -> "InventoryTransaction":
        del label
        raise InventoryStoreReadOnlyError(
            "This inventory-store view is read-only; request writable=True."
        )

    def fork(self, scenario_identity: Any = None) -> InventoryStore:
        return self._store.fork(scenario_identity)

    def checkpoint(self, path: str | Path) -> Path:
        return self._store.checkpoint(path)

    def iter_materialized(self, restore_metadata: bool = True):
        return self._store.iter_materialized(restore_metadata)


class InventoryTransaction:
    """Atomic mutation scope for an in-memory store."""

    def __init__(self, store: _InMemoryInventoryStore, label: str) -> None:
        self.store = store
        self.label = str(label)
        self._snapshot: _StoreState | None = None
        self._entered = False

    def __enter__(self) -> "InventoryTransaction":
        self.store._lock.acquire()
        if self.store._active_transaction:
            self.store._lock.release()
            raise InventoryStoreError(
                "Nested inventory transactions are not supported."
            )
        self.store._active_transaction = True
        self.store._ensure_owned_state()
        self._snapshot = self.store._transaction_snapshot()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if exc_type is not None:
                if self._snapshot is not None:
                    self.store._state = self._snapshot
            else:
                self.store._state.generation += 1
                self.store._state.transaction_log.append(self.label)
                if self.store.eager_indexes:
                    self.store._rebuild_indexes()
        finally:
            self.store._active_transaction = False
            self._entered = False
            self.store._lock.release()
        return False

    def _require_entered(self) -> None:
        if not self._entered:
            raise InventoryStoreError(
                "Transaction mutations must be called inside a with block."
            )

    def add_activity(self, payload: Mapping[str, Any]) -> ActivityId:
        self._require_entered()
        activity_id = self.store._add_activity_unchecked(payload)
        self.store._invalidate_indexes()
        return activity_id

    def clone_activity(
        self,
        source_id: ActivityId,
        activity_updates: Mapping[str, Any] | None = None,
        exchange_updates: (
            Mapping[int, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None
        ) = None,
    ) -> ActivityId:
        self._require_entered()
        payload = self.store._activity_payload(source_id)
        payload.update(copy.deepcopy(dict(activity_updates or {})))
        exchanges = payload.get("exchanges", [])
        if isinstance(exchange_updates, Mapping):
            for position, updates in exchange_updates.items():
                index = int(position)
                if index < 0 or index >= len(exchanges):
                    raise IndexError(f"Unknown source exchange position: {position}")
                exchanges[index].update(copy.deepcopy(dict(updates)))
        elif exchange_updates is not None:
            updates_list = list(exchange_updates)
            if len(updates_list) != len(exchanges):
                raise ValueError(
                    "Positional exchange_updates must match the source exchange count."
                )
            for exchange, updates in zip(exchanges, updates_list):
                exchange.update(copy.deepcopy(dict(updates)))
        activity_id = self.store._add_activity_unchecked(payload)
        self.store._invalidate_indexes()
        return activity_id

    def patch_activity(
        self,
        activity_id: ActivityId,
        updates: Mapping[str, Any],
        delete_fields: Iterable[str] = (),
    ) -> None:
        self._require_entered()
        if activity_id not in self.store._state.activities:
            raise KeyError(f"Unknown activity id: {activity_id}")
        changes = copy.deepcopy(dict(updates))
        exchanges = changes.pop("exchanges", None)
        payload = copy.deepcopy(self.store._state.activities[activity_id])
        payload.update(changes)
        for field_name in delete_fields:
            payload.pop(field_name, None)
        self.store._state.activities[activity_id] = payload
        if exchanges is not None:
            self.replace_exchanges(activity_id, exchanges)
        else:
            self.store._invalidate_indexes()

    def remove_activity(self, activity_id: ActivityId) -> None:
        self._require_entered()
        if activity_id not in self.store._state.activities:
            raise KeyError(f"Unknown activity id: {activity_id}")
        for exchange_id in list(self.store._state.activity_exchanges[activity_id]):
            self.store._remove_exchange_unchecked(exchange_id)
        del self.store._state.activity_exchanges[activity_id]
        del self.store._state.activities[activity_id]
        self.store._invalidate_indexes()

    def add_exchange(
        self, activity_id: ActivityId, payload: Mapping[str, Any]
    ) -> ExchangeId:
        self._require_entered()
        exchange_id = self.store._add_exchange_unchecked(activity_id, payload)
        self.store._invalidate_indexes()
        return exchange_id

    def patch_exchange(
        self,
        exchange_id: ExchangeId,
        updates: Mapping[str, Any],
        delete_fields: Iterable[str] = (),
    ) -> None:
        self._require_entered()
        if exchange_id not in self.store._state.exchanges:
            raise KeyError(f"Unknown exchange id: {exchange_id}")
        payload = copy.deepcopy(self.store._state.exchanges[exchange_id])
        payload.update(copy.deepcopy(dict(updates)))
        for field_name in delete_fields:
            payload.pop(field_name, None)
        self.store._state.exchanges[exchange_id] = payload
        self.store._invalidate_indexes()

    def remove_exchange(self, exchange_id: ExchangeId) -> None:
        self._require_entered()
        self.store._remove_exchange_unchecked(exchange_id)
        self.store._invalidate_indexes()

    def replace_exchanges(
        self, activity_id: ActivityId, exchanges: Iterable[Mapping[str, Any]]
    ) -> None:
        self._require_entered()
        if activity_id not in self.store._state.activities:
            raise KeyError(f"Unknown activity id: {activity_id}")
        for exchange_id in list(self.store._state.activity_exchanges[activity_id]):
            self.store._remove_exchange_unchecked(exchange_id)
        for exchange in exchanges:
            self.store._add_exchange_unchecked(activity_id, exchange)
        self.store._invalidate_indexes()


class InventoryStoreBuilder:
    """Incremental source-graph builder."""

    def __init__(self, backend: Literal["compact", "legacy"] = "compact") -> None:
        if backend not in {"compact", "legacy"}:
            raise ValueError("inventory backend must be 'compact' or 'legacy'")
        self.backend = backend
        self._store = create_inventory_store((), backend=backend)
        self._sealed = False

    def append(self, dataset: Mapping[str, Any]) -> None:
        if self._sealed:
            raise InventoryStoreError("Cannot append to a sealed builder.")
        self._store._add_activity_unchecked(dataset)

    def extend(self, datasets: Iterable[Mapping[str, Any]]) -> None:
        for dataset in datasets:
            self.append(dataset)

    def seal(self, scenario_identity: Any = None) -> InventoryStore:
        if self._sealed:
            raise InventoryStoreError(
                "InventoryStoreBuilder.seal() can only be called once."
            )
        self._sealed = True
        self._store._scenario_identity = scenario_identity
        if self._store.eager_indexes:
            self._store._rebuild_indexes()
        return self._store


def create_inventory_store(
    database: Iterable[Mapping[str, Any]],
    *,
    backend: Literal["compact", "legacy"] = "compact",
    scenario_identity: Any = None,
    take_ownership: bool = False,
) -> InventoryStore:
    if backend == "compact":
        return CompactInventoryStore(
            database,
            scenario_identity=scenario_identity,
            take_ownership=take_ownership,
        )
    if backend == "legacy":
        return LegacyInventoryStore(
            database,
            scenario_identity=scenario_identity,
            take_ownership=take_ownership,
        )
    raise ValueError("inventory_backend must be either 'compact' or 'legacy'.")


def get_scenario_inventory(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the private mutable working inventory for a sector wrapper.

    This bridge keeps the historical transformation implementations functional
    while preventing the mutable payload from being exposed as
    ``scenario["database"]``.  New transformation code should operate on the
    store contract directly.
    """

    if "_inventory_working_copy" in scenario:
        database = scenario["_inventory_working_copy"]
        if not isinstance(database, IndexedInventoryList):
            database = IndexedInventoryList(database)
            scenario["_inventory_working_copy"] = database
        return database
    if "database" in scenario:  # Compatibility for standalone legacy callers.
        return scenario["database"]
    store = scenario.get("_inventory_store")
    if store is None and scenario.get("_inventory_checkpoint") is not None:
        store = InventoryStore.open(scenario["_inventory_checkpoint"])
        scenario["_inventory_store"] = store
    if store is None:
        raise InventoryStoreError(
            "Scenario has no inventory store or private working inventory."
        )
    if isinstance(store, CompactInventoryStore):
        working_copy = store._checkout_materialized()
        scenario.pop("_inventory_store", None)
    else:
        working_copy = IndexedInventoryList(store.materialize(restore_metadata=True))
    scenario["_inventory_working_copy"] = working_copy
    return working_copy


def replace_scenario_inventory(
    scenario: dict[str, Any], database: Iterable[Mapping[str, Any]]
) -> None:
    """Commit a sector wrapper's resulting inventory without a public payload."""

    if "database" in scenario:
        scenario["database"] = database
        return
    store = scenario.get("_inventory_store")
    if store is None:
        scenario["_inventory_working_copy"] = IndexedInventoryList(database)
        return
    scenario["_inventory_store"] = create_inventory_store(
        database,
        backend=store.backend_name,
        scenario_identity=getattr(store, "scenario_identity", None),
    )
    scenario.pop("_inventory_working_copy", None)
    scenario.pop("_inventory_checkpoint", None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_table(path: Path, columns: Mapping[str, list[Any]]) -> None:
    if pa is None:
        with path.open("wb") as stream:
            pickle.dump(dict(columns), stream, protocol=pickle.HIGHEST_PROTOCOL)
        return
    arrays = {}
    for name, values in columns.items():
        array = pa.array(values)
        if pa.types.is_string(array.type):
            array = array.dictionary_encode()
        arrays[name] = array
    table = pa.table(arrays)
    with pa.OSFile(str(path), "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _read_table(path: Path) -> dict[str, list[Any]]:
    if pa is None:
        with path.open("rb") as stream:
            return pickle.load(stream)
    for open_reader in (pa_ipc.open_file, pa_ipc.open_stream):
        try:
            with pa.memory_map(str(path), "r") as source:
                table = open_reader(source).read_all()
            return {name: table[name].to_pylist() for name in table.column_names}
        except (pa.ArrowInvalid, pa.ArrowIOError):
            continue
    # A checkpoint produced in a source-only environment uses the pickle
    # fallback despite retaining the stable bundle filenames.
    with path.open("rb") as stream:
        return pickle.load(stream)


def _iter_table_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield Arrow rows batch-wise, with support for source-only fallbacks."""

    if pa is not None:
        try:
            with pa.memory_map(str(path), "r") as source:
                reader = pa_ipc.open_file(source)
                for batch_index in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_index)
                    names = batch.schema.names
                    columns = [
                        batch.column(index).to_pylist() for index in range(len(names))
                    ]
                    for values in zip(*columns):
                        yield dict(zip(names, values))
            return
        except (pa.ArrowInvalid, pa.ArrowIOError):
            pass
        try:
            with pa.memory_map(str(path), "r") as source:
                reader = pa_ipc.open_stream(source)
                for batch in reader:
                    names = batch.schema.names
                    columns = [
                        batch.column(index).to_pylist() for index in range(len(names))
                    ]
                    for values in zip(*columns):
                        yield dict(zip(names, values))
            return
        except (pa.ArrowInvalid, pa.ArrowIOError):
            pass
    columns = _read_table(path)
    names = tuple(columns)
    for values in zip(*(columns[name] for name in names)):
        yield dict(zip(names, values))


def _exchange_from_arrow_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for field_name in _EXCHANGE_STRING_FIELDS:
        value = row.get(field_name)
        if value is not None:
            payload[field_name] = value
    category_0 = row.get("categories__0")
    category_1 = row.get("categories__1")
    if category_0 is not None and category_1 is not None:
        payload["categories"] = (category_0, category_1)
    for field_name in _EXCHANGE_NUMERIC_FIELDS:
        kind = int(row.get(f"{field_name}__kind") or _NUMERIC_MISSING)
        if kind != _NUMERIC_MISSING:
            payload[field_name] = _decode_numeric_column(
                kind,
                row.get(f"{field_name}__float"),
                row.get(f"{field_name}__int"),
            )
    for field_name in _EXCHANGE_BOOLEAN_FIELDS:
        value = row.get(field_name)
        if value is not None:
            payload[field_name] = bool(value)
    return payload


def _write_checkpoint_payloads(
    store: _InMemoryInventoryStore,
    temporary: Path,
    *,
    batch_size: int = 65_536,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], list[str]]:
    """Serialize metadata and Arrow rows in one ordered graph traversal."""

    id_columns = ("exchange_id", "activity_id", "exchange_ordinal")
    category_columns = ("categories__0", "categories__1")
    numeric_columns = tuple(
        column
        for field_name in _EXCHANGE_NUMERIC_FIELDS
        for column in (
            f"{field_name}__kind",
            f"{field_name}__float",
            f"{field_name}__int",
        )
    )
    column_names = (
        *id_columns,
        *_EXCHANGE_STRING_FIELDS,
        *category_columns,
        *numeric_columns,
        *_EXCHANGE_BOOLEAN_FIELDS,
    )

    strings: list[str] = []
    seen_strings: set[str] = set()

    def collect_string(value: Any) -> None:
        if isinstance(value, str) and value not in seen_strings:
            strings.append(value)
            seen_strings.add(value)

    def append_payload(columns, payload) -> dict[str, Any]:
        for field_name in _EXCHANGE_STRING_FIELDS:
            value = payload.get(field_name)
            columns[field_name].append(value if isinstance(value, str) else None)
            if pa is None:
                collect_string(value)
        categories = payload.get("categories")
        valid_categories = (
            isinstance(categories, tuple)
            and len(categories) == 2
            and all(isinstance(item, str) for item in categories)
        )
        columns["categories__0"].append(categories[0] if valid_categories else None)
        columns["categories__1"].append(categories[1] if valid_categories else None)
        if pa is None and valid_categories:
            collect_string(categories[0])
            collect_string(categories[1])
        numeric_kinds = {}
        for field_name in _EXCHANGE_NUMERIC_FIELDS:
            kind, float_value, int_value = _numeric_column_parts(
                payload.get(field_name, _UNHASHABLE)
            )
            numeric_kinds[field_name] = kind
            columns[f"{field_name}__kind"].append(kind)
            columns[f"{field_name}__float"].append(float_value)
            columns[f"{field_name}__int"].append(int_value)
        for field_name in _EXCHANGE_BOOLEAN_FIELDS:
            value = payload.get(field_name)
            columns[field_name].append(value if type(value) is bool else None)
        return _exchange_sidecar_metadata(payload, numeric_kinds=numeric_kinds)

    schema = None
    if pa is not None:
        dictionary_type = pa.dictionary(pa.int32(), pa.string())
        fields = [
            pa.field("exchange_id", pa.int64()),
            pa.field("activity_id", pa.int64()),
            pa.field("exchange_ordinal", pa.int64()),
        ]
        fields.extend(
            pa.field(field_name, dictionary_type)
            for field_name in (*_EXCHANGE_STRING_FIELDS, *category_columns)
        )
        fields.extend(
            pa.field(
                column,
                (
                    pa.int8()
                    if column.endswith("__kind")
                    else pa.int64() if column.endswith("__int") else pa.float64()
                ),
            )
            for column in numeric_columns
        )
        fields.extend(
            pa.field(field_name, pa.bool_()) for field_name in _EXCHANGE_BOOLEAN_FIELDS
        )
        schema = pa.schema(fields)
    columns = {name: [] for name in column_names}

    def flush(writer) -> None:
        if not columns["exchange_id"]:
            return
        assert schema is not None
        arrays = []
        for field in schema:
            values = columns[field.name]
            if pa.types.is_dictionary(field.type):
                array = pa.array(values, type=pa.string()).dictionary_encode()
                for value in array.dictionary.to_pylist():
                    collect_string(value)
                arrays.append(array)
            else:
                arrays.append(pa.array(values, type=field.type))
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        for values in columns.values():
            values.clear()

    offsets = {"kind": [], "id": [], "offset": [], "length": []}
    activities: dict[str, list[Any]] = defaultdict(list)

    def serialize(sidecar, writer=None) -> None:
        exchange_cursor = 0
        for ordinal, activity_id in enumerate(store.iter_activity_ids()):
            payload = store._state.activities[activity_id]
            exchange_ids = store._state.activity_exchanges[activity_id]
            exchange_metadata = []
            for exchange_ordinal, exchange_id in enumerate(exchange_ids):
                exchange_payload = store._state.exchanges[exchange_id]
                columns["exchange_id"].append(exchange_id)
                columns["activity_id"].append(activity_id)
                columns["exchange_ordinal"].append(exchange_ordinal)
                metadata = append_payload(columns, exchange_payload)
                if metadata:
                    exchange_metadata.append((exchange_ordinal, metadata))
                if writer is not None and len(columns["exchange_id"]) >= batch_size:
                    flush(writer)

            bundle = {
                "activity": payload,
                "exchange_metadata": exchange_metadata,
            }
            encoded = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
            offset = sidecar.tell()
            sidecar.write(encoded)
            offsets["kind"].append("activity-bundle")
            offsets["id"].append(activity_id)
            offsets["offset"].append(offset)
            offsets["length"].append(len(encoded))
            activities["activity_id"].append(activity_id)
            activities["source_ordinal"].append(ordinal)
            exchange_count = len(exchange_ids)
            activities["exchange_start"].append(exchange_cursor)
            activities["exchange_count"].append(exchange_count)
            exchange_cursor += exchange_count
            for field_name in _ACTIVITY_COMMON_FIELDS:
                value = payload.get(field_name)
                activities[field_name].append(value if isinstance(value, str) else None)
                collect_string(value)

        if writer is not None:
            flush(writer)

    with (temporary / "metadata.bin").open("wb") as sidecar:
        if pa is None:
            serialize(sidecar)
        else:
            with pa.OSFile(str(temporary / "exchanges.arrow"), "wb") as sink:
                # IPC streams permit independent dictionaries per bounded batch,
                # avoiding a second graph traversal to build a global dictionary.
                with pa_ipc.new_stream(sink, schema) as writer:
                    serialize(sidecar, writer)

    if pa is None:
        _write_table(temporary / "exchanges.arrow", columns)

    return offsets, dict(activities), strings


def _write_checkpoint(store: _InMemoryInventoryStore, path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == Path(path.anchor):
        raise ValueError("Refusing to use a filesystem root as a checkpoint path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    backup: Path | None = None
    try:
        offsets, activities, strings = _write_checkpoint_payloads(store, temporary)

        _write_table(temporary / "strings.arrow", {"value": strings})
        _write_table(temporary / "activities.arrow", dict(activities))
        _write_table(temporary / "metadata_offsets.arrow", offsets)

        try:
            from . import __version__ as premise_version

            premise_version_text = ".".join(map(str, premise_version))
        except (ImportError, TypeError):  # pragma: no cover - defensive fallback
            premise_version_text = "unknown"

        manifest = {
            "schema_version": STORE_SCHEMA_VERSION,
            "premise_version": premise_version_text,
            "backend": store.backend_name,
            "scenario_identity": store.scenario_identity,
            "generation": store.generation,
            "activity_count": len(store),
            "exchange_count": len(store._state.exchanges),
            "row_counts": {
                "activities": len(store),
                "exchanges": len(store._state.exchanges),
            },
            "source_fingerprint": _sha256(temporary / "metadata.bin"),
            "inventory_fingerprints": [],
            "uncertainty_settings": {},
            "columnar_format": "arrow-ipc" if pa is not None else "pickle-fallback",
            "metadata_layout": "activity-extras-v4",
        }
        try:
            manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
        except TypeError:
            manifest["scenario_identity"] = repr(store.scenario_identity)
            manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
        (temporary / "manifest.json").write_text(manifest_text, encoding="utf-8")

        checksummed = [
            "manifest.json",
            "strings.arrow",
            "activities.arrow",
            "exchanges.arrow",
            "metadata.bin",
            "metadata_offsets.arrow",
        ]
        checksums = {name: _sha256(temporary / name) for name in checksummed}
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, sort_keys=True, indent=2), encoding="utf-8"
        )

        if path.exists():
            backup = path.with_name(f".{path.name}.old-{os.getpid()}")
            if backup.exists():
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            os.replace(path, backup)
        os.replace(temporary, path)
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        return path
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists() and not path.exists():
            os.replace(backup, path)
        raise


def _open_checkpoint(path: Path) -> InventoryStore:
    path = path.expanduser().resolve()
    manifest_path = path / "manifest.json"
    checksums_path = path / "checksums.json"
    if not manifest_path.exists() or not checksums_path.exists():
        raise InventoryStoreCorruptionError(
            f"Inventory checkpoint at {path} is incomplete."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryStoreCorruptionError(
            f"Cannot read inventory checkpoint at {path}."
        ) from error
    if manifest.get("schema_version") != STORE_SCHEMA_VERSION:
        raise InventoryStoreVersionError(
            "Unsupported inventory-store schema "
            f"{manifest.get('schema_version')!r}; expected {STORE_SCHEMA_VERSION}."
        )
    for name, expected in checksums.items():
        candidate = path / name
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise InventoryStoreCorruptionError(
                f"Checksum validation failed for checkpoint file {name!r}."
            )

    offsets = _read_table(path / "metadata_offsets.arrow")
    bundles: dict[int, dict[str, Any]] = {}
    with (path / "metadata.bin").open("rb") as sidecar:
        for kind, record_id, offset, length in zip(
            offsets["kind"], offsets["id"], offsets["offset"], offsets["length"]
        ):
            if kind != "activity-bundle":
                raise InventoryStoreCorruptionError(
                    f"Invalid metadata record kind {kind!r} for schema "
                    f"{STORE_SCHEMA_VERSION}."
                )
            sidecar.seek(offset)
            encoded = sidecar.read(length)
            try:
                bundle = pickle.loads(encoded)
            except Exception as error:
                raise InventoryStoreCorruptionError(
                    f"Invalid metadata payload for {kind} {record_id}."
                ) from error
            if not isinstance(bundle, dict) or not {
                "activity",
                "exchange_metadata",
            }.issubset(bundle):
                raise InventoryStoreCorruptionError(
                    f"Invalid activity bundle for activity {record_id}."
                )
            bundles[record_id] = bundle

    activities = _read_table(path / "activities.arrow")
    database: list[dict[str, Any]] = []
    datasets_by_id = {}
    exchange_metadata_by_id = {}
    for activity_id in activities.get("activity_id", []):
        try:
            bundle = bundles[activity_id]
        except KeyError as error:
            raise InventoryStoreCorruptionError(
                f"Missing activity bundle for activity {activity_id}."
            ) from error
        payload = bundle["activity"]
        payload["exchanges"] = []
        database.append(payload)
        datasets_by_id[activity_id] = payload
        exchange_metadata_by_id[activity_id] = dict(bundle["exchange_metadata"])

    exchange_count = 0
    for row in _iter_table_rows(path / "exchanges.arrow"):
        activity_id = row["activity_id"]
        try:
            dataset = datasets_by_id[activity_id]
        except KeyError as error:
            raise InventoryStoreCorruptionError(
                f"Exchange references missing activity {activity_id}."
            ) from error
        exchange_ordinal = int(row["exchange_ordinal"])
        if exchange_ordinal != len(dataset["exchanges"]):
            raise InventoryStoreCorruptionError(
                f"Non-contiguous exchange order for activity {activity_id}."
            )
        exchange = _exchange_from_arrow_row(row)
        exchange.update(exchange_metadata_by_id[activity_id].get(exchange_ordinal, {}))
        dataset["exchanges"].append(exchange)
        exchange_count += 1

    if exchange_count != manifest.get("exchange_count"):
        raise InventoryStoreCorruptionError(
            "Exchange row count does not match the checkpoint manifest."
        )

    backend = manifest.get("backend", "compact")
    store = create_inventory_store(
        database,
        backend=backend if backend in {"compact", "legacy"} else "compact",
        scenario_identity=manifest.get("scenario_identity"),
        take_ownership=True,
    )
    store._state.generation = int(manifest.get("generation", 0))
    return store


__all__ = [
    "ActivityId",
    "ExchangeId",
    "ActivityKey",
    "ProviderKey",
    "FilterExpression",
    "ActivityQuery",
    "ActivityRecord",
    "ExchangeRecord",
    "InventoryStore",
    "InventoryTransaction",
    "InventoryStoreBuilder",
    "LegacyInventoryStore",
    "CompactInventoryStore",
    "ReadOnlyInventoryStore",
    "IndexedInventoryList",
    "InventoryStoreError",
    "InventoryStoreReadOnlyError",
    "InventoryStoreCorruptionError",
    "InventoryStoreVersionError",
    "STORE_SCHEMA_VERSION",
    "create_inventory_store",
    "get_scenario_inventory",
    "replace_scenario_inventory",
    "install_wurst_query_engine",
    "get_wurst_query_diagnostics",
]
