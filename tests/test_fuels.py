from types import SimpleNamespace

from premise.fuels.base import Fuels


def test_gcam_coal_methane_inventory_is_regionalized():
    coal_methane = {
        "name": (
            "methane, synthetic, gaseous, 5 bar, from coal-based hydrogen, "
            "at fuelling station"
        ),
        "reference product": "methane, high pressure",
        "location": "RER",
    }
    fuels = object.__new__(Fuels)
    fuels.database = [coal_methane]
    fuels.fuel_map = {"methane, from coal": [coal_methane]}
    fuels.iam_data = SimpleNamespace(
        production_volumes=None,
        natural_gas_blend=None,
    )
    fuels.mapping = SimpleNamespace(generate_fuel_map=lambda: {})

    captured = {}

    def capture_regionalization(mapping, production_volumes):
        captured.update(mapping)

    fuels.process_and_add_activities = capture_regionalization

    fuels.generate_biogas_activities()

    assert captured == {"methane, from coal": [coal_methane]}


def test_fuel_carbon_update_preserves_ordered_exchange_semantics():
    fuel_input = {
        "name": "market for test fuel",
        "product": "test fuel",
        "location": "GLO",
        "unit": "kilogram",
        "type": "technosphere",
        "amount": 2.0,
    }
    fossil = {
        "name": "Carbon dioxide, fossil",
        "unit": "kilogram",
        "type": "biosphere",
        "amount": 10.0,
    }
    consumer = {
        "name": "fuel consumer",
        "location": "R1",
        "exchanges": [fuel_input, fossil],
    }
    market = {
        "name": "market for test fuel",
        "location": "R1",
        "exchanges": [fuel_input.copy(), fossil.copy()],
    }
    fuels = object.__new__(Fuels)
    fuels.database = [consumer, market]
    fuels.fuel_map = {"biofuel": [], "fossil": []}
    fuels.iam_data = SimpleNamespace(production_volumes=None)
    fuels.regions = ["R1"]
    fuels.ecoinvent_to_iam_loc = {}
    fuels.biosphere_flows = {
        ("Carbon dioxide, non-fossil", "air", "unspecified", "kilogram"): "flow"
    }
    fuels.is_in_index = lambda exchange, location: location == "R1"
    fuels.get_technology_and_regional_production_shares = lambda **kwargs: (
        None,
        {("biofuel", "R1"): 0.5, ("fossil", "R1"): 0.5},
        {"R1": 1.0},
    )

    fuels.update_fuel_carbon_dioxide_emissions(
        variables=["biofuel", "fossil"],
        market_names=["market for test fuel"],
        co2_intensity=1.0,
        fossil_variables=["fossil"],
    )

    assert fuel_input["location"] == "R1"
    assert fossil["amount"] == 9.0
    assert consumer["exchanges"][-1]["name"] == "Carbon dioxide, non-fossil"
    assert consumer["exchanges"][-1]["amount"] == 1.0
    assert len(market["exchanges"]) == 2
    assert market["exchanges"][0]["location"] == "GLO"
    assert market["exchanges"][1]["amount"] == 10.0
