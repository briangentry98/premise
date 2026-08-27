import numpy as np

from premise.geomap import Geomap
from premise.inventory_imports import canonicalize_classification_key
from premise.validation import (
    BaseDatasetValidator,
    DatasetNormalizer,
    normalize_exact_deterministic_exchange_duplicates,
    normalize_inventory_numeric_types,
    normalize_inventory_uncertainty,
)


def _validator_for_locations(database_locations, regions=None, extra_regions=None):
    validator = object.__new__(BaseDatasetValidator)
    validator.original_database = [{"location": "GLO"}]
    validator.database = [{"location": location} for location in database_locations]
    validator.regions = regions or []
    validator.valid_regions = set(validator.regions) | set(extra_regions or [])
    validator.geo = Geomap("remind")
    validator.major_issues_log = []
    validator.minor_issues_log = []
    return validator


def test_check_new_location_accepts_extra_superstructure_regions():
    validator = _validator_for_locations(
        database_locations=["JAP"],
        regions=["JPN"],
        extra_regions=["JAP"],
    )

    validator.check_new_location()

    assert validator.major_issues_log == []


def test_check_new_location_logs_unregistered_location_as_major_issue():
    validator = _validator_for_locations(
        database_locations=["not-a-location"],
        regions=["JPN"],
    )

    validator.check_new_location()

    assert len(validator.major_issues_log) == 1
    assert validator.major_issues_log[0]["location"] == "not-a-location"


def test_expected_iam_location_does_not_remap_an_iam_region():
    validator = _validator_for_locations(database_locations=["ME"], regions=["ME"])

    assert validator.expected_iam_location("ME") == "ME"


def test_export_normalizer_adds_missing_classifications(tmp_path, monkeypatch):
    dataset = {
        "name": "fuel cell system assembly, 1 kWe, proton exchange membrane (PEM)",
        "reference product": "fuel cell system, 1 kWe, proton exchange membrane (PEM)",
        "location": "GLO",
        "classifications": [],
        "exchanges": [],
    }
    expected = [
        (
            "ISIC rev.4 ecoinvent",
            "4322:Plumbing, heat and air-conditioning installation",
        ),
        ("CPC", "46410: Primary cells and primary batteries"),
    ]

    normalizer = object.__new__(DatasetNormalizer)
    normalizer.database = [dataset]
    normalizer.classifications = {
        canonicalize_classification_key(
            dataset["name"], dataset["reference product"]
        ): {
            "ISIC rev.4 ecoinvent": expected[0][1],
            "CPC": expected[1][1],
        }
    }

    monkeypatch.chdir(tmp_path)

    normalizer.add_missing_classifications()

    assert dataset["classifications"] == expected


def test_numeric_normalization_converts_scalar_arrays_but_not_vectors():
    scalar = np.array(0.25)
    singleton = np.array([2.0])
    vector = np.array([1.0, 2.0])
    database = [
        {
            "name": "activity",
            "log parameters": {"share": scalar},
            "exchanges": [
                {"amount": scalar, "loc": singleton},
                {"amount": vector},
            ],
        }
    ]

    normalize_inventory_numeric_types(database)

    assert database[0]["log parameters"]["share"] == 0.25
    assert database[0]["exchanges"][0]["amount"] == 0.25
    assert database[0]["exchanges"][0]["loc"] == 2.0
    assert database[0]["exchanges"][1]["amount"] is vector


def test_uncertainty_normalization_repairs_lognormal_sign_metadata():
    database = [
        {
            "exchanges": [
                {
                    "amount": -2.0,
                    "uncertainty type": 2,
                    "loc": np.log(2.0),
                    "scale": 0.2,
                    "negative": False,
                }
            ]
        }
    ]

    normalize_inventory_uncertainty(database)

    assert database[0]["exchanges"][0]["negative"] is True


def test_exact_deterministic_duplicates_are_summed_but_stochastic_rows_remain():
    deterministic = {
        "name": "market for fuel",
        "product": "fuel",
        "location": "World",
        "unit": "kilogram",
        "type": "technosphere",
        "amount": 0.25,
        "uncertainty type": 0,
    }
    stochastic = {
        **deterministic,
        "amount": 0.5,
        "uncertainty type": 2,
        "loc": np.log(0.5),
        "scale": 0.2,
    }
    database = [
        {
            "exchanges": [
                dict(deterministic),
                dict(deterministic),
                dict(stochastic),
                dict(stochastic),
            ]
        }
    ]

    normalize_exact_deterministic_exchange_duplicates(database)

    assert len(database[0]["exchanges"]) == 3
    assert database[0]["exchanges"][0]["amount"] == 0.5
    assert database[0]["exchanges"][1:] == [stochastic, stochastic]
