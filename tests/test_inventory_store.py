import json
import inspect
import random

import numpy as np
import pytest
from wurst import searching as ws

import premise.new_database as new_database_module
import premise.utils as utils_module
from premise.inventory_store import (
    ActivityKey,
    ActivityQuery,
    CompactInventoryStore,
    FilterExpression,
    InventoryStore,
    InventoryStoreBuilder,
    InventoryStoreCorruptionError,
    InventoryStoreError,
    InventoryStoreReadOnlyError,
    InventoryStoreVersionError,
    IndexedInventoryList,
    LegacyInventoryStore,
    ProviderKey,
    ReadOnlyInventoryStore,
    get_scenario_inventory,
    replace_scenario_inventory,
)
from premise.new_database import NewDatabase


@pytest.fixture
def inventory():
    return [
        {
            "name": "market for electricity, low voltage",
            "reference product": "electricity, low voltage",
            "location": "CH",
            "unit": "kilowatt hour",
            "database": "ecoinvent",
            "code": "ch-grid",
            "classifications": [("CPC", "171")],
            "custom": {"tuple": (1, 2), "list": [np.float64(3.5)]},
            "exchanges": [
                {
                    "name": "market for electricity, low voltage",
                    "product": "electricity, low voltage",
                    "location": "CH",
                    "unit": "kilowatt hour",
                    "type": "production",
                    "amount": 1.0,
                    "output": ("ecoinvent", "ch-grid"),
                }
            ],
        },
        {
            "name": "market for electricity, low voltage",
            "reference product": "electricity, low voltage",
            "location": "CH",
            "unit": "kilowatt hour",
            "database": "ecoinvent",
            "code": "ch-grid-duplicate",
            "exchanges": [],
        },
        {
            "name": "electricity use",
            "reference product": "service",
            "location": "CH",
            "unit": "unit",
            "database": "ecoinvent",
            "code": "consumer",
            "exchanges": [
                {
                    "name": "market for electricity, low voltage",
                    "product": "electricity, low voltage",
                    "location": "CH",
                    "unit": "kilowatt hour",
                    "type": "technosphere",
                    "amount": np.float64(2.0),
                    "input": ("ecoinvent", "ch-grid"),
                }
            ],
        },
        {
            "name": "market for electricity, low voltage",
            "reference product": "electricity, low voltage",
            "location": "GLO",
            "unit": "kilowatt hour",
            "database": "ecoinvent",
            "code": "glo-grid",
            "exchanges": [],
        },
    ]


@pytest.mark.parametrize("store_class", [LegacyInventoryStore, CompactInventoryStore])
def test_ordered_queries_duplicates_masks_and_immutable_records(store_class, inventory):
    store = store_class(inventory)
    query = ActivityQuery(
        filters=(
            FilterExpression("name", "market for", "contains"),
            FilterExpression("name", "market", "startswith"),
        ),
        masks=(FilterExpression("location", "GLO"),),
    )

    records = store.find(query)

    assert [record["code"] for record in records] == ["ch-grid", "ch-grid-duplicate"]
    assert store.contains(
        ActivityKey(
            "market for electricity, low voltage",
            "electricity, low voltage",
            "CH",
        )
    )
    with pytest.raises(TypeError):
        records[0]["custom"]["new"] = "forbidden"
    with pytest.raises(AttributeError):
        records[0]["custom"]["list"].append(4)


def test_compact_indexes_are_lazy_and_invalidated_atomically(inventory):
    store = CompactInventoryStore(inventory)
    assert store._state.indexes_ready is False
    assert store._state.exchange_owner == {}
    assert type(store._state.exchanges).__name__ == "_DenseExchangeTable"
    assert isinstance(store._state.activity_exchanges[0], range)

    assert store.find_one({"code": "consumer"})["name"] == "electricity use"
    assert store._state.indexes_ready is True

    with store.transaction("move-consumer") as tx:
        tx.patch_activity(2, {"location": "DE"})
        assert store._state.indexes_ready is False

    assert store.find_one({"location": "DE"})["code"] == "consumer"
    assert store._state.indexes_ready is True

    exchange = store.exchange(0)
    assert exchange.activity_id == 0
    assert len(store._state.exchange_owner) == len(store._state.exchanges)

    with store.transaction("append-exchange") as tx:
        tx.add_exchange(0, {"name": "new flow", "type": "biosphere", "amount": 1})
    assert isinstance(store._state.activity_exchanges[0], list)


def test_find_one_provider_order_and_reverse_consumers(inventory):
    store = LegacyInventoryStore(inventory)
    provider_key = ProviderKey(
        "market for electricity, low voltage",
        "electricity, low voltage",
        "kilowatt hour",
    )

    assert [record["location"] for record in store.providers(provider_key, "CH")] == [
        "CH",
        "CH",
        "GLO",
    ]
    assert store.consumers(0) == (2,)
    with pytest.raises(ValueError, match="found 2"):
        store.find_one({"location": "CH", "unit": "kilowatt hour"})
    assert store.find_one({"code": "consumer"}).id == 2


@pytest.mark.parametrize("store_class", [LegacyInventoryStore, CompactInventoryStore])
def test_transaction_commands_commit_and_rollback(store_class, inventory):
    store = store_class(inventory)
    initial = store.materialize()

    with pytest.raises(RuntimeError):
        with store.transaction("rollback") as tx:
            tx.patch_activity(0, {"location": "DE"})
            tx.remove_exchange(0)
            tx.add_activity(
                {
                    "name": "temporary",
                    "reference product": "temporary",
                    "location": "GLO",
                    "unit": "unit",
                    "exchanges": [],
                }
            )
            raise RuntimeError("abort")

    assert store.materialize() == initial
    assert store.generation == 0

    with store.transaction("complete") as tx:
        cloned = tx.clone_activity(
            0,
            {"name": "cloned grid", "code": "clone"},
            {0: {"amount": 4.0}},
        )
        tx.patch_activity(cloned, {"comment": "added"}, delete_fields=("custom",))
        exchange_id = tx.add_exchange(
            cloned,
            {"name": "flow", "type": "biosphere", "amount": 1.0},
        )
        tx.patch_exchange(exchange_id, {"amount": 2.0})
        tx.remove_exchange(exchange_id)
        tx.replace_exchanges(
            1,
            [{"name": "replacement", "type": "production", "amount": 1.0}],
        )
        tx.remove_activity(3)

    assert store.generation == 1
    assert store.find_one({"code": "clone"})["exchanges"][0]["amount"] == 4.0
    assert (
        store.find_one({"code": "ch-grid-duplicate"})["exchanges"][0]["name"]
        == "replacement"
    )
    assert not store.contains(
        ActivityKey(
            "market for electricity, low voltage",
            "electricity, low voltage",
            "GLO",
        )
    )


def test_nested_and_out_of_context_transactions_are_rejected(inventory):
    store = LegacyInventoryStore(inventory)
    transaction = store.transaction("outside")
    with pytest.raises(InventoryStoreError, match="with block"):
        transaction.add_activity({})

    with store.transaction("outer"):
        with pytest.raises(InventoryStoreError, match="Nested"):
            with store.transaction("inner"):
                pass


def test_compact_forks_are_copy_on_write_and_isolated(inventory):
    source = CompactInventoryStore(inventory)
    forks = [source.fork(identity) for identity in ("a", "b", "c")]

    with forks[0].transaction("scenario:a") as tx:
        tx.patch_activity(0, {"location": "DE"})
    with forks[1].transaction("scenario:b") as tx:
        tx.remove_activity(0)
    with forks[2].transaction("scenario:c") as tx:
        tx.add_activity(
            {
                "name": "scenario c",
                "reference product": "service",
                "location": "GLO",
                "unit": "unit",
                "exchanges": [],
            }
        )

    assert source.activity(0)["location"] == "CH"
    assert forks[0].activity(0)["location"] == "DE"
    assert len(forks[1]) == len(source) - 1
    assert len(forks[2]) == len(source) + 1


def test_compact_checkout_requires_exclusive_or_discarded_shared_state(inventory):
    source = CompactInventoryStore(inventory)
    scenario = source.fork("scenario")

    with pytest.raises(InventoryStoreError, match="shared compact state"):
        scenario._checkout_materialized()

    checked_out = scenario._checkout_materialized(discard_shared_state=True)

    assert checked_out == inventory
    assert len(scenario) == 0
    checked_out[0]["location"] = "DE"
    # A forced checkout transfers ownership away from every shared reference;
    # NewDatabase drops the source reference as part of the same operation.
    assert source.activity(0)["location"] == "DE"


@pytest.mark.parametrize("store_class", [LegacyInventoryStore, CompactInventoryStore])
def test_checkpoint_roundtrip_preserves_arbitrary_metadata(
    store_class, inventory, tmp_path
):
    inventory[2]["exchanges"][0].update(
        {
            "uncertainty type": 2,
            "loc": np.float32(-0.25),
            "production volume": 42.5,
            "categories": ("air", "urban air close to ground"),
            "comment": "kept in the lossless sidecar",
            "maximum": None,
        }
    )
    store = store_class(inventory, scenario_identity=("image", 2050))
    with store.transaction("metadata") as tx:
        tx.patch_activity(0, {"empty": None, "categories": ("air", "urban")})

    checkpoint = store.checkpoint(tmp_path / "scenario.inventory-store")
    reopened = InventoryStore.open(checkpoint)

    materialized = reopened.materialize()
    exchange = materialized[2]["exchanges"][0]

    assert reopened.backend_name == store.backend_name
    assert materialized == store.materialize()
    assert type(exchange["amount"]) is np.float64
    assert type(exchange["uncertainty type"]) is int
    assert type(exchange["loc"]) is np.float32
    assert type(exchange["production volume"]) is float
    assert exchange["categories"] == ("air", "urban air close to ground")
    assert exchange["comment"] == "kept in the lossless sidecar"
    assert exchange["maximum"] is None
    assert set(path.name for path in checkpoint.iterdir()) == {
        "manifest.json",
        "strings.arrow",
        "activities.arrow",
        "exchanges.arrow",
        "metadata.bin",
        "metadata_offsets.arrow",
        "checksums.json",
    }


def test_checkpoint_corruption_and_schema_are_rejected(inventory, tmp_path):
    checkpoint = CompactInventoryStore(inventory).checkpoint(
        tmp_path / "scenario.inventory-store"
    )
    (checkpoint / "metadata.bin").write_bytes(b"corrupt")
    with pytest.raises(InventoryStoreCorruptionError, match="Checksum"):
        InventoryStore.open(checkpoint)

    checkpoint = CompactInventoryStore(inventory).checkpoint(checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InventoryStoreVersionError, match="Unsupported"):
        InventoryStore.open(checkpoint)


def test_legacy_pickle_cleanup_preserves_inventory_checkpoints(
    inventory, monkeypatch, tmp_path
):
    checkpoint = CompactInventoryStore(inventory).checkpoint(
        tmp_path / "scenario.inventory-store"
    )
    legacy_pickle = tmp_path / "scenario.pickle"
    legacy_pickle.write_bytes(b"legacy")
    monkeypatch.setattr(utils_module, "DIR_CACHED_FILES", tmp_path)

    utils_module.delete_all_pickles()

    assert checkpoint.is_dir()
    assert not legacy_pickle.exists()


def test_builder_and_read_only_facade(inventory):
    builder = InventoryStoreBuilder("compact")
    builder.extend(inventory[:2])
    store = builder.seal("scenario")
    view = ReadOnlyInventoryStore(store)

    assert len(view) == 2
    with pytest.raises(InventoryStoreReadOnlyError):
        view.transaction("forbidden")
    with pytest.raises(InventoryStoreError, match="only be called once"):
        builder.seal()


def test_internal_ids_never_appear_in_materialized_payloads(inventory):
    store = CompactInventoryStore(inventory)

    assert all("activity_id" not in dataset for dataset in store.materialize())
    assert all(
        "exchange_id" not in exchange
        for dataset in store.materialize()
        for exchange in dataset["exchanges"]
    )


def test_new_database_public_store_api_and_removed_database_attribute(inventory):
    obj = object.__new__(NewDatabase)
    obj._inventory_api_active = True
    obj.inventory_backend = "compact"
    obj._source_inventory_store = CompactInventoryStore(inventory)
    obj.scenarios = [{"model": "image", "pathway": "SSP2-Base", "year": 2050}]

    with pytest.raises(AttributeError, match="materialize_inventory"):
        _ = obj.database
    with pytest.raises(AttributeError, match="get_inventory_store"):
        obj.database = []

    view = obj.get_inventory_store()
    assert isinstance(view, ReadOnlyInventoryStore)
    assert obj.materialize_inventory() == inventory
    assert "database" not in obj.scenarios[0]

    writable = obj.get_inventory_store(writable=True)
    with writable.transaction("test") as tx:
        tx.patch_activity(0, {"location": "DE"})
    assert obj.materialize_inventory()[0]["location"] == "DE"
    assert obj._source_inventory_store.materialize()[0]["location"] == "CH"


def test_compact_backend_stays_opt_in_until_performance_gate_passes():
    parameter = inspect.signature(NewDatabase).parameters["inventory_backend"]
    assert parameter.default == "legacy"


def test_new_database_runtime_materialization_is_not_attached_to_scenario(inventory):
    obj = object.__new__(NewDatabase)
    obj._inventory_api_active = True
    obj.inventory_backend = "compact"
    obj._source_inventory_store = CompactInventoryStore(inventory)
    obj.scenarios = [{"model": "image", "pathway": "SSP2-Base", "year": 2050}]

    runtime = obj._load_scenario_database_for_update(obj.scenarios[0], 0)

    assert runtime["_inventory_working_copy"] == inventory
    assert "database" not in obj.scenarios[0]
    runtime["_inventory_working_copy"][0]["location"] = "DE"
    obj._store_updated_scenario(obj.scenarios[0], runtime, persist=False)
    assert "database" not in obj.scenarios[0]
    assert obj.materialize_inventory()[0]["location"] == "DE"


def test_new_database_compact_final_scenario_checks_out_reloadable_source(inventory):
    obj = object.__new__(NewDatabase)
    obj._inventory_api_active = True
    obj.inventory_backend = "compact"
    obj._source_inventory_store = CompactInventoryStore(inventory)
    obj.scenarios = [{"model": "image", "pathway": "SSP2-Base", "year": 2050}]
    obj.database_cache_filepath = "source-cache"
    obj.inventories_cache_filepath = "inventory-cache"
    obj.additional_inventories = None

    runtime = obj._load_scenario_database_for_update(obj.scenarios[0], 0)

    assert runtime["_inventory_working_copy"] == inventory
    assert obj._source_inventory_store is None
    assert "_inventory_store" not in obj.scenarios[0]


@pytest.mark.parametrize("persist", [False, True])
def test_new_database_update_keeps_only_private_inventory_state(
    inventory, monkeypatch, tmp_path, persist
):
    obj = object.__new__(NewDatabase)
    obj._inventory_api_active = True
    obj.inventory_backend = "compact"
    obj._source_inventory_store = CompactInventoryStore(inventory)
    obj.scenarios = [{"model": "image", "pathway": "SSP2-Base", "year": 2050}]
    obj.version = "3.12"
    obj.system_model = "cutoff"
    obj.use_absolute_efficiency = False
    obj.gains_scenario = "CLE"
    obj.database_cache_filepath = None
    obj.inventories_cache_filepath = None
    obj.additional_inventories = None

    def fake_update(scenario, version, system_model):
        assert version == "3.12"
        assert system_model == "cutoff"
        assert "database" not in scenario
        database = get_scenario_inventory(scenario)
        database.append(
            {
                "name": "updated activity",
                "reference product": "service",
                "location": "GLO",
                "unit": "unit",
                "code": "updated",
                "exchanges": [],
            }
        )
        replace_scenario_inventory(scenario, database)
        return scenario

    monkeypatch.setattr(new_database_module, "_update_biomass", fake_update)
    monkeypatch.setattr(new_database_module, "DIR_CACHED_FILES", tmp_path)

    obj.update("biomass", persist=persist)

    scenario = obj.scenarios[0]
    assert "database" not in scenario
    assert "_inventory_working_copy" not in scenario
    assert ("_inventory_checkpoint" in scenario) is persist
    assert ("_inventory_store" in scenario) is (not persist)
    assert obj.get_inventory_store().find_one({"code": "updated"})["name"] == (
        "updated activity"
    )


def test_legacy_and_compact_match_after_randomized_mutations(inventory):
    rng = random.Random(42)
    rng_values = [rng.random() for _ in range(40)]
    stores = [LegacyInventoryStore(inventory), CompactInventoryStore(inventory)]

    for step in range(40):
        activity_ids = list(stores[0].iter_activity_ids())
        operation = rng.choice(("patch", "clone", "add_exchange", "replace"))
        target = rng.choice(activity_ids)
        for store in stores:
            with store.transaction(f"random:{step}:{operation}") as tx:
                if operation == "patch":
                    tx.patch_activity(target, {"random value": rng_values[step]})
                elif operation == "clone":
                    tx.clone_activity(
                        target,
                        {
                            "name": f"clone {step}",
                            "code": f"random-clone-{step}",
                        },
                    )
                elif operation == "add_exchange":
                    tx.add_exchange(
                        target,
                        {
                            "name": f"flow {step}",
                            "type": "biosphere",
                            "amount": rng_values[step],
                        },
                    )
                else:
                    tx.replace_exchanges(
                        target,
                        [
                            {
                                "name": f"replacement {step}",
                                "type": "production",
                                "amount": rng_values[step],
                            }
                        ],
                    )
        assert stores[0].materialize() == stores[1].materialize()


def test_indexed_wurst_bridge_preserves_order_and_invalidates(inventory):
    database = IndexedInventoryList(inventory)
    filters = (
        ws.contains("name", "market for"),
        ws.equals("unit", "kilowatt hour"),
        ws.exclude(ws.equals("location", "GLO")),
    )

    assert [dataset["code"] for dataset in ws.get_many(database, *filters)] == [
        "ch-grid",
        "ch-grid-duplicate",
    ]

    database.append(
        {
            "name": "market for added product",
            "reference product": "added product",
            "location": "CH",
            "unit": "kilowatt hour",
            "code": "added",
            "exchanges": [],
        }
    )
    assert [dataset["code"] for dataset in ws.get_many(database, *filters)] == [
        "ch-grid",
        "ch-grid-duplicate",
        "added",
    ]


def test_indexed_wurst_bridge_uses_ordered_fallback_for_dynamic_predicates(inventory):
    database = IndexedInventoryList(inventory)
    dynamic = lambda dataset: dataset.get("code", "").endswith("grid")

    assert [dataset["code"] for dataset in ws.get_many(database, dynamic)] == [
        "ch-grid",
        "glo-grid",
    ]


def test_indexed_wurst_bridge_preserves_equals_none_semantics(inventory):
    database = IndexedInventoryList(inventory)

    assert [
        dataset["code"] for dataset in ws.get_many(database, ws.equals("type", None))
    ] == [
        "ch-grid",
        "ch-grid-duplicate",
        "consumer",
        "glo-grid",
    ]
