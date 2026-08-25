# Runtime and resident-memory optimization

> **Premise 3.0 InventoryStore status (24 August 2026):** The store contract,
> immutable records, atomic transactions, copy-on-write forks, ordered indexes,
> Arrow checkpoints, private scenario ownership, and explicit materialization
> API are implemented. The compact backend is still opt-in. On the current
> warm IMAGE SSP2-M 2050 all-sector differential run it produced the exact same
> semantic hash as the legacy store (`39efcf273da6b52e6c6bdfce8a6546a9a0819a054340fee9062ac2a7fddf808c`),
> with 50,938 datasets and 1,597,616 exchanges. A matched diagnostic run (not
> the five-run acceptance median) reduced wall time from 82.67 to 45.25 seconds
> (45.3%) and sampled peak RSS from 2.692 to 2.030 GB (24.6%). This is well
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
| End-to-end wall time | 82.67 s | 45.25 s | -45.3% |
| Sampled peak RSS | 2.692 GB | 2.030 GB | -24.6% |
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

Shared proxy creation now uses a memoized inventory-aware clone. Mutable
dictionaries, lists, and NumPy arrays remain isolated, shared nested references
remain shared within a clone, and uncommon objects retain the generic
`deepcopy` fallback. Immutable Python and NumPy scalars are reused. The same
profile also found that mining-share loading filtered and normalized a temporary
DataFrame but returned the untouched source DataFrame; removing that dead work
preserved the exact source hash. Together these changes reduced the metals
sector from 9.96 to 8.13 seconds and the full update from 56.83 to 54.65 seconds
in matched compact runs.

Provider resolution now caches the ordered provider-location grouping for each
name/product key and invalidates those groupings whenever a sector adds or
removes a provider. GIS match decisions use a reserved scenario-level cache, so
identical geographic queries are reused across sector transformation objects
without changing candidate or tie-breaking order. In the next matched compact
run, metals fell from 8.13 to 7.50 seconds, the full update from 54.65 to 53.01
seconds, and end-to-end time from 57.14 to 55.73 seconds. The checkpoint retained
the exact seed-zero canonical hash and counts reported below.

GIS matching also caches the synthetic Rest-of-World face set for each ordered
provider-location tuple within a transformation. Different consumer locations
can reuse that topology while still running the legacy contained/intersects
selection independently. The cache owns the set created by the geography
resolver directly and is released with the sector object; only the much smaller
final GIS decisions persist between sectors. Metals fell further from 7.50 to
6.22 seconds, the full update to 51.74 seconds, and end-to-end time to 54.34
seconds. Sampled peak RSS was 1.945 GB in this run.

Exact provider-location membership checks now reuse the same generation-aware
location set as provider grouping instead of rebuilding a location list for
every call. Provider additions and removals invalidate both views atomically.
This removed repeated list construction from roughly 605,000 metals calls and
also accelerated several other sectors. The next diagnostic reached 50.30
seconds for the update and 52.83 seconds end to end. Its absolute sampled peak
was 1.956 GB; the process started 22 MB above the preceding run, while its
constructor-to-peak growth was 22 MB lower. The strict checkpoint hash and
counts remained exact.

The inventory-aware proxy clone now recognizes common immutable scalar types
through exact-type lookup and reuses ordinary immutable dictionary keys without
recursively dispatching through the clone function. NumPy scalars still retain
their exact type and identity, mutable containers and arrays remain isolated,
and uncommon key/value types retain the generic `deepcopy` fallback. This cut
millions of recursive clone calls. A second matched diagnostic reached 50.13
seconds for the update, 5.93 seconds for metals, 52.65 seconds end to end, and a
1.941 GB sampled peak, with the exact canonical hash and counts preserved.

Metals post-allocation correction now builds one sparse index of activities
with kilogram in-ground natural-resource exchanges. Its mapping retains the
original exchange dictionaries, stores no entries for the overwhelmingly common
empty case, and is rebuilt before each correction pass. Nine correction helpers
reuse it while standalone helper calls keep their scan fallback. This reduced
the next matched diagnostic to 49.86 seconds for the update, 5.73 seconds for
metals, and 52.37 seconds end to end, with a 1.941 GB sampled peak and exact
canonical output.

Metals transport initialization now builds one country/metal-to-row-position
mapping instead of two 23,345-row `iterrows()` dictionaries containing full
`Series` objects. The unused duplicate mapping is gone, duplicate keys retain
the previous last-row precedence, and lookups select the row directly instead
of scanning every key. Metals validation also reuses the already-cached mining
share frame rather than reopening the Excel sheet. Metals fell from 5.73 to
4.74 seconds, the full update to 48.98 seconds, and end-to-end time to 51.59
seconds. Metals-end RSS was about 28 MB lower; the later sampled process peak
was 1.945 GB. Canonical output remained exact.

Fuel-efficiency matching now compiles each electricity technology's sanitized
fuel names into an immutable sorted prefix index. A binary search preserves the
legacy `filter.startswith(exchange_name)` rule while avoiding millions of
repeated string replacements and Python generator iterations; accepted
exchanges and their numerical reduction order remain unchanged. In the scoped
profile, `find_fuel_efficiency` fell from 3.04 to 0.13 seconds and the enclosing
efficiency update from 3.36 to 1.05 seconds. The matched unprofiled run reduced
electricity from 4.59 to 3.53 seconds, the full update to 47.24 seconds, and
end-to-end time to 49.77 seconds. Sampled peak RSS was 1.935 GB, and the strict
canonical hash and counts remained exact.

Coal-power-plant adjustment now caches the scalar result of each xarray
selection by country, fuel, CHP status, and variable, as well as each derived
emission factor. NumPy scalars retain the original multiplication and division
order without holding thousands of zero-dimensional xarray objects alive.
Selections fell from 4,833 to 3,281 and the profiled coal adjustment from 1.56
to 0.56 seconds. In the next matched unprofiled run, electricity reached 3.24
seconds, the full update 45.94 seconds, and end-to-end time 48.51 seconds.
Sampled peak RSS was 1.938 GB, within 3 MB of the preceding run, and strict
canonical output remained exact.

Common relinking now aggregates amounts for each exact exchange identity once,
in original exchange order, instead of rescanning all technosphere inputs for
every unique exchange. Exact provider membership checks inline the existing
generation-aware per-key location cache, avoiding repeated helper and generator
dispatch while retaining incremental invalidation after index mutations. In the
scoped profile, relinking fell from 3.21 to 1.54 seconds,
`find_new_exchange_entries` from 1.81 to 0.50 seconds, and `is_in_index` from
1.04 to 0.43 seconds. The matched unprofiled run reduced electricity from 3.24
to 2.79 seconds, the full update to 44.98 seconds, and end-to-end time to 47.58
seconds. Sampled peak RSS remained 1.938 GB and strict canonical output was
exact.

Mapping-file loading now keeps a one-entry cache of parsed YAML content so
adjacent variable-specific requests reuse the same document without retaining
all mapping files for the lifetime of the build. In the scoped electricity
profile, full YAML parses fell from five to two, `get_mapping` from 0.45 to 0.14
seconds, and electricity initialization from 1.41 to 1.09 seconds. A repeated
unprofiled run put electricity at 2.73 seconds versus 2.79 seconds previously;
the 47.62-second end-to-end result was within noise of the 47.58-second
headline, which is therefore unchanged. Strict canonical output remained
exact.

Indexed Wurst candidate construction now returns cached immutable sets without
copying them for every query, caches compound exclusions and unions, and starts
multi-filter intersections with the smallest candidate group. The scoped query
kernel fell from 0.73 to 0.56 seconds. Electricity market construction also
reuses each region's supplier selection and production-volume shares across
period-specific low- and high-voltage markets. This removed 4.47 million
repeated location comparisons and reduced high-voltage regional-market
construction from 0.73 to 0.38 seconds. Both changes retained the exact
canonical output.

Exact provider membership now uses a reference-counted semantic index keyed by
name, product, and location. Provider additions and removals update it
incrementally, including duplicate providers, while the existing ordered
provider groupings retain generation-based invalidation. Common relinking binds
one semantic snapshot and filters technosphere exchanges directly, avoiding a
provider helper call for every exchange. In the scoped profile,
`relink_datasets` fell from 1.51 to 1.10 seconds and `is_in_index` from 0.42 to
0.12 seconds. The matched unprofiled run reduced electricity from 2.73 to 2.27
seconds, the full update from 45.02 to 42.72 seconds, and end-to-end time from
47.62 to 45.25 seconds. Its 2.030 GB sampled peak was 87 MB above the preceding
single run, so this diagnostic is a runtime improvement but not evidence of an
RSS improvement. Strict canonical output remained exact.

Activity-map name prefiltering now preserves `IndexedInventoryList` for compact
working inventories instead of materializing the reduced candidates into a
plain list. The 212 downstream contains/mask queries therefore continue through
the indexed query engine; ordinary list callers retain the legacy scan. In the
scoped electricity profile, activity-map construction fell from 0.79 to 0.62
seconds, inner filtering from 0.60 to 0.44 seconds, and the sector from 5.36 to
5.04 seconds. Indexed queries increased by the expected 634 without adding a
fallback, and strict canonical output remained exact.

Electricity efficiency updating now caches each IAM scalar by technology and
IAM region for the duration of one update. Datasets sharing that pair reuse the
same value while the general DataArray helper remains uncached for callers that
may mutate their input. IAM efficiency selections fell from 1,280 to 407, their
cost from 0.26 to 0.08 seconds, and the profiled efficiency update from 0.77 to
0.57 seconds. The unprofiled electricity phase fell from 2.27 to 2.14 seconds.
Unrelated sectors were slower in that single run, putting end-to-end time at
46.62 rather than 45.25 seconds, so the documented headline remains unchanged.
Strict canonical output remained exact.

Transport initialization no longer binds the incoming provider index to the
relinking-cache argument. That positional mismatch retained the previous full
provider-index generation while building a second one for every transport
mode. The first consecutive transport now deliberately rebuilds its index,
because earlier sectors may have changed indexed fields directly; subsequent
transport modes reuse that freshly maintained index until a non-transport
sector runs. The six transport phases fell from 9.72 to 9.26 seconds in the
matched compact diagnostics, and the duplicate provider-index generation is no
longer retained as scenario cache. Absolute RSS was affected by an unrelated
macOS allocator purge during the verification run, so it is not used as a new
memory headline. The exact canonical hash and inventory counts remained
unchanged.

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
