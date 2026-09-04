Premise 2.5 release and migration guide
=======================================

Version 2.5.1 moves runtime metals rules to validated YAML, applies each
dataset/rule pair once, preserves the component-based material structure of
``EPR construction``, and records rule-specific decisions and target values in
the structured change report. The inventory API introduced in 2.5.0 is
unchanged.

Version 2.5.0 introduces a controlled inventory API, read-only validation
certificates, structured change reports, and a more efficient scenario and
export pipeline. Constructor, update, and export signatures remain available,
but integrations that accessed the mutable ``NewDatabase.database`` attribute
must migrate.

Inventory access
----------------

After initialization, inventories are owned by an ``InventoryStore``. Read
operations return immutable snapshots:

.. code-block:: python

    ndb = NewDatabase(...)
    store = ndb.get_inventory_store()
    activities = store.find({"location": "CH"})

Mutations must be explicit and atomic. Request a writable store and make all
changes inside a transaction:

.. code-block:: python

    store = ndb.get_inventory_store(writable=True)
    with store.transaction("custom:foreground") as transaction:
        transaction.patch_activity(
            activities[0].id,
            {"comment": "Updated by the foreground integration"},
        )

If an integration cannot consume the store API, it can request an independent
``list[dict]``:

.. code-block:: python

    database = ndb.materialize_inventory(restore_metadata=True)

Materialization duplicates the full activity and exchange graph and can require
several gigabytes for ecoinvent. It should therefore remain an integration
boundary rather than the normal inspection path.

The ``inventory_backend`` constructor argument accepts ``"compact"`` and
``"legacy"``. Both implement the same contract. The compact backend adds
copy-on-write scenario forks, indexed queries, and versioned Arrow checkpoints;
the legacy backend remains available for compatibility and differential
testing.

Validation certificates
-----------------------

Every completed update receives a read-only methodological certificate.
Unsuppressed errors stop checkpointing or export; warnings and documented,
versioned suppressions remain visible:

.. code-block:: python

    report = ndb.get_validation_report(scenario=0)
    report.raise_for_errors()

    # Run the complete inventory-graph diagnostic when required.
    exhaustive = ndb.get_validation_report(scenario=0, exhaustive=True)

Production validation combines sector contracts for transformation coverage,
finite physical values, reference production, market composition, supplier
links, and sector-specific methodological expectations. Exporters reuse that
certificate and add their own schema-validation phase.

Structured change reports
-------------------------

The former pipe-delimited workbook has been replaced. Reports now consist of a
review-oriented Excel workbook and a complete Parquet audit:

.. code-block:: python

    ndb = NewDatabase(..., generate_reports=False)
    ndb.update()
    artifacts = ndb.generate_change_report(
        filepath="review/reports",
        name="ssp2-review.xlsx",
    )
    print(artifacts.workbook_path)
    print(artifacts.details_path)

``generate_change_report()`` returns an immutable ``ChangeReportArtifacts``
instance. Setting ``generate_reports=False`` disables automatic reports after
exports, but does not disable this explicit method or methodological
validation. Existing report files are never overwritten or migrated.

Update and export workflow
--------------------------

``update_and_write()`` avoids the former intermediate scenario dump and reload
when a scenario should be written directly to Brightway:

.. code-block:: python

    ndb.update_and_write(name="image-ssp2-2050")

For exploratory use, the repository now provides a numbered series of 12
notebooks covering construction, consequential scenarios, custom inputs,
external datapackages, export formats, scenario arrays, matrices, incremental
databases, reports, and score comparison.

Further details
---------------

See :doc:`structured_change_report` for the report schemas and lifecycle. The
complete release notes, including performance, diesel-market, battery-market,
and certification changes, are recorded in the project ``CHANGELOG.md``.
