# Runtime and resident-memory optimization

> **Premise 3.0 InventoryStore status (24 August 2026):** The store contract,
> immutable records, atomic transactions, copy-on-write forks, ordered indexes,
> Arrow checkpoints, private scenario ownership, and explicit materialization
> API are implemented. The compact backend is still opt-in. On the current
> warm IMAGE SSP2-M 2050 all-sector differential run it produced the exact same
> semantic hash as the legacy store (`39efcf273da6b52e6c6bdfce8a6546a9a0819a054340fee9062ac2a7fddf808c`),
> with 50,938 datasets and 1,597,616 exchanges. A matched diagnostic run (not
> the five-run acceptance median) reduced wall time from 82.67 to 59.34 seconds
> (28.2%) and sampled peak RSS from 2.692 to 1.972 GB (26.7%). This is well
> short of the required 50%/50% activation gate.
> The default therefore remains `legacy`; transformation hot paths still need
> to move off their private mutable working materialization.

This document records the profiling baseline, the first optimization pass, and
the architectural work that should follow it. The benchmark is a complete
`NewDatabase.update()` over all sectors for one ecoinvent 3.12 cutoff scenario.
It stops after premise writes its internal scenario cache; it does not mutate a
Brightway database.

## Reproducible benchmark

Run the benchmark with the project environment and an existing Brightway
project. Encrypted IAM files require `PREMISE_KEY` or `IAM_FILES_KEY`; a local
plaintext IAM file does not.

```console
PYTHONHASHSEED=0 python benchmarks/profile_new_database.py \
  --model remind --pathway SSP1-NPi \
  --inventory-backend compact \
  --output /tmp/premise-profile.json
```

Add `--pstats /tmp/premise-profile.pstats` for a deterministic `cProfile`
capture. RSS is sampled every 50 ms with `psutil` and cross-checked against
`resource.getrusage`.

## Results

### Compact InventoryStore pass

The current IMAGE SSP2-M 2050 diagnostic uses
`inventory_backend="compact"` explicitly and fixes `PYTHONHASHSEED=0`. It
includes the full all-sector update and compact checkpoint write.

| Metric | Legacy oracle | Compact | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall time | 82.67 s | 59.34 s | -28.2% |
| Sampled peak RSS | 2.692 GB | 1.972 GB | -26.7% |
| Datasets | 50,938 | 50,938 | exact |
| Exchanges | 1,597,616 | 1,597,616 | exact |

The compact checkpoint is 348 MB. Common exchange strings and numeric values
are dictionary-encoded or typed in batched Arrow record batches; arbitrary
metadata remains in a lossless per-activity sidecar. Python `float` and `int`
and NumPy `float32` and `float64` values retain their exact scalar types when a
checkpoint is reopened. Source and final indexes are lazy, exchange storage is
dense and ordered, and the final single-scenario build transfers exclusive
ownership instead of deep-copying the working graph.

Emissions is the first sector migrated off the private mutable working list. A
compact build promotes the graph to a store before the final emissions update,
compiles the legacy GAINS contains/mask mapping against activity metadata, and
patches only affected activities and exchanges in one atomic transaction. The
sector fell from 4.27 to 2.90 seconds in the matched full build. Compact
transaction rollback now snapshots graph structure and replaces touched row
payloads, avoiding a transaction-wide deep copy while retaining exact rollback
semantics.

The seed-zero canonical hash is
`39efcf273da6b52e6c6bdfce8a6546a9a0819a054340fee9062ac2a7fddf808c`.
Compact builds refuse an unfixed hash seed in the benchmark harness so an
existing order-sensitive transformation cannot be mistaken for a backend
regression.

### Earlier mutable-inventory optimization pass

The comparison below used Python 3.11 on macOS, a warm source-database cache,
REMIND SSP1-NPi in 2050, and all sectors. Both revisions used the same source
database, IAM file, process sampler, and cache state. These are the final
results after the strict output-equivalence fixes; earlier, faster measurements
from the metadata-elision experiment are intentionally not reported as valid.

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall time | 89.60 s | 86.82 s | -3.1% |
| `update()` wall time | 87.06 s | 83.58 s | -4.0% |
| Sampled peak RSS | 1.818 GB | 1.795 GB | -1.3% |
| Constructor-end RSS | 1.612 GB | 1.614 GB | +0.1% |
| Metals sector | 30.67 s | 28.15 s | -8.2% |
| Emissions sector | 7.16 s | 4.70 s | -34.3% |
| Scenario-cache dump | 8.37 s | 5.99 s | -28.5% |

An additional IMAGE SSP2-M baseline exposed the same structure at a larger
scale: 119.99 seconds without profiling, 2.46 billion calls under `cProfile`,
and 1.61 GB warm-cache peak RSS. Its metals update alone took 41.65 seconds.

Cold-cache peak RSS is not materially improved by this pass. Brightway source
extraction must still materialize the complete, metadata-rich database before
premise can compact it. The new cache format improves subsequent builds and is
versioned so old and compact caches cannot be confused.

Activity counts were identical after every sector in the baseline and optimized
runs (42,374 after emissions), and the optimized validation log contained no
major issues. The stronger end-to-end equivalence certification is recorded
below.

## Output-equivalence certification

`benchmarks/compare_build_outputs.py` builds the parent revision and optimized
revision in separate processes with the same fixed `PYTHONHASHSEED`, source
database, encrypted IAM input, model, pathway, year, and system model. It then:

1. restores all scenario-cache metadata and computes a canonical SHA-256 over
   every dataset and exchange field except Brightway's random storage IDs;
2. writes each result to a dedicated Brightway database, extracts it again, and
   repeats the exact canonical comparison;
3. compares stable Brightway database metadata exactly; and
4. compares 20 LCIA scores (five representative activities and four methods)
   with relative and absolute tolerances of `1e-12`.

The full IMAGE SSP2-M 2050 all-sector certification against revision
`691ec672` passed. Both sides contained 50,938 datasets and 1,597,616 exchanges.
The restored-scenario hash was
`1aa05bbf2d84ed5e58ba787b54d1705a582f1a000af60eb490897f87b4e0be79`;
the re-extracted Brightway hash was
`4482604d947aeaa08c99366fc10b6693e669ea33fd4666d5e3b16c42b87ec4ac`.
Stable metadata and all 20 LCIA scores also matched.

Example invocation (run once from each revision with different output and
database names):

```console
PYTHONHASHSEED=0 PREMISE_KEY=... python benchmarks/compare_build_outputs.py build \
  --output-dir /tmp/equivalence/baseline \
  --label baseline --revision 691ec672 \
  --inventory-backend compact \
  --database-name premise-equivalence-baseline

python benchmarks/compare_build_outputs.py compare \
  --left /tmp/equivalence/baseline \
  --right /tmp/equivalence/optimized \
  --report /tmp/equivalence/report.json
```

## Profile findings

The main baseline costs were structural rather than numerical:

- The database held roughly 29,400 activities and 963,000 exchanges before
  scenario expansion. Source activity comments alone contained about 63
  million characters and remained resident during every sector update.
- Activity-map construction accumulated 49 seconds of profiled time in repeated
  Wurst predicate scans.
- Metals accumulated 63 seconds of profiled time. Mining-share filters rescanned
  the full activity list for each Excel row, even though the ecoinvent 3.12
  mapping is almost entirely exact name/product matching.
- Scenario-cache creation walked each exchange twice: once to collect sidecar
  metadata and again to trim it. It called the exchange trimmer about 3.2
  million times in the profiled run.
- General-purpose `is_in_index` and activity-existence checks repeatedly built
  temporary location lists or scanned the database.

## Implemented changes

### Versioned source cache

Cache filenames include a schema version so a layout change forces one safe
rebuild instead of loading an incompatible cache. Source comments remain in the
runtime database: removing them changed proxy comments and therefore failed the
strict canonical equivalence gate.

### One-pass scenario compaction

Scenario metadata extraction and exchange trimming now happen in one traversal,
which removes a complete pass over roughly 1.6 million exchanges. Cache-value
semantics remain identical to the parent revision.

### Indexed metals matching

Mining-share workbooks are parsed once per normalized ecoinvent version and
callers receive defensive DataFrame copies. Exact `equals`/`either` expressions
are resolved through the existing name/reference-product activity index. More
general `contains`, `startswith`, and boolean expressions retain a compatible
scan fallback. Mining dataset membership is cached after the first lookup. The
index is refreshed once after regional proxy creation; the equivalence harness
caught the stale-index version because it left five mineral-resource exchanges
different from the parent revision.

### Safer activity-map prefiltering

Activity maps prefilter once on deduplicated names only when every mapping entry
actually has a name constraint. This fixes string filters being expanded into
characters and preserves mappings based only on product, unit, or location.

## Experiments deliberately not retained

- Calling `gc.collect` and allocator-specific page-release hooks after every
  sector saved only about 7 MB while adding roughly 4 seconds.
- Replacing all Wurst filters with nested Python predicate calls reduced CPU in
  isolated cases but increased allocator-retained RSS.
- Adding broad secondary indexes to every `BaseTransformation` made ownership
  and mutation synchronization more complex and produced unstable RSS. The
  first pass therefore indexes only a measured, exact metals lookup.

## Next architectural steps

These changes are deliberately larger and should be developed behind benchmark
and equivalence gates.

1. **Copy-on-write scenario database.** Keep an immutable compact source store
   and represent each scenario as changed activities plus tombstones. Materialize
   ordinary dictionaries only at an exporter boundary. This removes the largest
   source of duplicated dicts, strings, and exchange lists.
2. **Compact exchange representation.** Store common exchange fields in typed or
   columnar arrays and uncommon metadata in a side table. Nearly one million
   small exchange dictionaries dominate object overhead.
3. **Unified mutation-aware activity index.** Own exact name/product/location,
   name-token, and reverse-consumer indexes in one component. Require all sector
   code to mutate through its API so indexes cannot drift. This would replace
   the remaining Wurst scans and repeated consumer searches safely.
4. **Process-isolated sector pipeline.** Run selected high-water sectors
   serially in short-lived worker processes with cache checkpoints. The OS then
   reclaims arenas between phases; failures are isolated. Serialization cost
   must be measured against the RSS ceiling.
5. **Streaming metadata restoration and export.** Avoid restoring every comment,
   classification, and uncommon exchange field before writing. Join sidecar
   shards in bounded chunks directly into Brightway, SimaPro, and openLCA
   exporters.
6. **Compiled inventory mappings.** Validate YAML/Excel filters at package build
   time and compile exact version-specific lookups. Keep runtime predicates only
   for genuinely dynamic filters.
7. **Delta-oriented multi-scenario execution.** Share the immutable source and
   IAM-independent indexes across scenarios. Parallelize scenarios only with a
   configured RSS budget; concurrent sector updates would otherwise multiply
   the current high-water mark.

Every stage should preserve activity/exchange counts, linking diagnostics,
scenario metadata, and representative LCIA checks in addition to unit tests.
