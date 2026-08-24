# InventoryStore API (premise 3.0)

Premise 3.0 owns inventories through `InventoryStore`. Mutable inventory lists
are no longer exposed on `NewDatabase` or retained in active scenario
dictionaries.

```python
ndb = premise.NewDatabase(
    scenarios=[{"model": "image", "pathway": "SSP2-M", "year": 2050}],
    source_db="ecoinvent-3.12-cutoff",
    inventory_backend="compact",  # opt-in while the performance gate is open
)

store = ndb.get_inventory_store()
activities = store.find(
    ActivityQuery((FilterExpression("location", "CH"),))
)

with ndb.get_inventory_store(writable=True).transaction("custom:foreground") as tx:
    tx.patch_activity(activities[0].id, {"comment": "updated"})
```

Read methods return immutable snapshots. Additions, patches, cloning, removals,
and exchange replacement must occur inside a transaction. A transaction commits
data and index changes together and restores its complete prior state if an
exception leaves the context.

Compact transactions keep a rollback-safe structural snapshot and copy only
rows they touch. The emissions transformation uses this path directly; sectors
not yet migrated continue through the private list-compatible bridge.

Integrations that cannot consume the store may explicitly materialize it:

```python
database = ndb.materialize_inventory(restore_metadata=True)
```

This duplicates the complete graph as Python dictionaries and can require
several gigabytes for a full ecoinvent scenario. It is therefore an integration
boundary, not the normal inspection API.

## Backends and checkpoints

`legacy` is the dictionary-backed semantic oracle. `compact` provides
copy-on-write scenario forks, ordered indexes, and versioned Arrow IPC
checkpoints with a lossless metadata sidecar. Common exchange strings and
numeric values use typed, batched Arrow columns; arbitrary fields are stored in
one sidecar bundle per activity. Reopening preserves Python and NumPy numeric
scalar types exactly. The bundle contains:

```text
manifest.json
strings.arrow
activities.arrow
exchanges.arrow
metadata.bin
metadata_offsets.arrow
checksums.json
```

Checkpoint writes use a sibling temporary directory and replacement; every
file is verified before a bundle is opened. Store schema versions are
independent from the historical pickle cache schema.

The compact backend remains opt-in until every required single- and
multi-scenario benchmark reduces both wall time and sampled peak RSS by at
least 50%. `legacy` remains the constructor default while this release gate is
open.
