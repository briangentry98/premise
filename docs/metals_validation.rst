Metals rules and validation
===========================

Runtime configuration
---------------------

``premise/data/metals/metal_products.yaml`` is the authoritative material-rule
file. ``technology_conversion_factors.yaml`` contains activity unit conversion
factors. Both files use ``schema_version: 1`` and are parsed once into validated,
immutable rule objects. Invalid identifiers, selectors, factors, application
modes, or allocation groups stop the update before inventory mutation.

``set_direct_amount`` replaces matching direct technosphere inputs. Biosphere
exchanges are never selected by this operation. Providers are matched by exact
name and reference product, with the location preference ``World``, ``GLO``,
then ``RoW``. If no configured provider exists, the rule is skipped and a
validation warning and provenance decision are recorded.

EPR construction
----------------

The ``preserve-epr-material-structure`` policy skips plant-level material
overlays for every ``EPR construction`` location. EPR contains detailed child
components which already demand several configured metals. Lead and tin have
explicit upstream component paths; this is evidence of overlap for those
products, but it does not prove that every configured material is present.

The structured report records applied rules with
``metals.material_rule.applied`` and preserved EPR rules with
``metals.material_rule.preserved_source``. Actual exchange differences appear
under ``Key Changes``. Preserve decisions, including their rule and policy IDs,
appear under ``Methodology``.

Dataset-wise comparison
-----------------------

``dev/compare_metal_inputs.py`` compares an unupdated source database, an
updated pre-fix database and an updated fixed database. It aggregates configured
direct metal inputs by dataset and product, while retaining supplier details in
the output, and writes complete and unexpected-only CSV files. A release is
accepted only when
``unexpected_differences.csv`` has no data rows.

The decision list is available after the metals update as
``ndb.scenarios[0]["mapping"]["metals"]["material decisions"]``. Save that
list as JSON for the comparison and overlap tools.

Example::

    python dev/compare_metal_inputs.py \
        --project ecoinvent-3.11-cutoff \
        --source SOURCE_DB \
        --before BEFORE_FIX_DB \
        --after AFTER_FIX_DB \
        --decisions material-decisions.json \
        --output results/premise_251_metals_validation

``dev/audit_metal_rule_overlaps.py`` follows exact provider-product paths to a
bounded depth. Its upstream totals are a screening result, not an
installed-material balance. The audit is deliberately separate from
``NewDatabase.update()`` and therefore adds no production runtime.
