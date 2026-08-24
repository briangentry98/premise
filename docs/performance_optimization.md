# Runtime and resident-memory optimization

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
python benchmarks/profile_new_database.py \
  --model remind --pathway SSP1-NPi \
  --output /tmp/premise-profile.json
```

Add `--pstats /tmp/premise-profile.pstats` for a deterministic `cProfile`
capture. RSS is sampled every 50 ms with `psutil` and cross-checked against
`resource.getrusage`.

## Results

The comparison below used Python 3.11 on macOS, a warm source-database cache,
REMIND SSP1-NPi in 2050, and all sectors. Both revisions used the same source
database, IAM file, process sampler, and cache state.

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall time | 89.60 s | 72.33 s | -19.3% |
| `update()` wall time | 87.06 s | 69.82 s | -19.8% |
| Sampled peak RSS | 1.818 GB | 1.706 GB | -6.2% |
| Constructor-end RSS | 1.612 GB | 1.531 GB | -5.0% |
| Metals sector | 30.67 s | 21.25 s | -30.7% |
| Emissions sector | 7.16 s | 4.72 s | -34.1% |
| Scenario-cache dump | 8.37 s | 7.54 s | -9.9% |

An additional IMAGE SSP2-M baseline exposed the same structure at a larger
scale: 119.99 seconds without profiling, 2.46 billion calls under `cProfile`,
and 1.61 GB warm-cache peak RSS. Its metals update alone took 41.65 seconds.

Cold-cache peak RSS is not materially improved by this pass. Brightway source
extraction must still materialize the complete, metadata-rich database before
premise can compact it. The new cache format improves subsequent builds and is
versioned so old and compact caches cannot be confused.

Activity counts were identical after every sector in the baseline and optimized
runs (42,374 after emissions), and the optimized validation log contained no
major issues.

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

### Compact, versioned source cache

Dataset comments now live in the metadata sidecar instead of the hot source
database. They are restored before export, after which scenario-generated
comments are merged in the same order as before. The heat-pump comments needed
by CDR classification remain resident because they are runtime input, not only
export metadata. Cache filenames include a schema version to force one safe
rebuild instead of loading an incompatible old cache.

### One-pass scenario compaction

Scenario metadata extraction and exchange trimming now happen in one traversal.
Fast scalar paths in cache-value detection avoid routing millions of ordinary
integers and floats through pandas.

### Indexed metals matching

Mining-share workbooks are parsed once per normalized ecoinvent version and
callers receive defensive DataFrame copies. Exact `equals`/`either` expressions
are resolved through the existing name/reference-product activity index. More
general `contains`, `startswith`, and boolean expressions retain a compatible
scan fallback. Mining dataset membership is cached after the first lookup.

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
