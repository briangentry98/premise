import argparse
import os
from pathlib import Path

import gcamreader


def resolve_query_scenario(
    conn: gcamreader.LocalDBConn, requested_scenario: str | None
) -> tuple[str, str]:
    scenarios_in_db = conn.listScenariosInDB()

    if scenarios_in_db is None or scenarios_in_db.empty:
        raise ValueError("No scenarios were found in the GCAM database.")

    if requested_scenario is None:
        selected = scenarios_in_db.iloc[-1]
        print(
            "No scenario was specified; defaulting to the most recent scenario "
            f"in the database: {selected['fqName']}"
        )
        return selected["fqName"], selected["name"]

    fqname_matches = scenarios_in_db.loc[
        scenarios_in_db["fqName"] == requested_scenario
    ]
    if len(fqname_matches) == 1:
        selected = fqname_matches.iloc[0]
        return selected["fqName"], requested_scenario

    name_matches = scenarios_in_db.loc[scenarios_in_db["name"] == requested_scenario]
    if len(name_matches) == 1:
        selected = name_matches.iloc[0]
        return selected["fqName"], requested_scenario

    if len(name_matches) > 1:
        available = "\n  - ".join(name_matches["fqName"].tolist())
        raise ValueError(
            "Multiple scenarios matched the requested name. Use the fully "
            f"qualified scenario name instead:\n  - {available}"
        )

    available = "\n  - ".join(scenarios_in_db["fqName"].tolist())
    raise ValueError(
        f"Scenario '{requested_scenario}' was not found in the database. "
        f"Available scenarios are:\n  - {available}"
    )


def run_queries(
    dbfile: str,
    scenario: str | None = None,
    dbpath: str = "database",
    query_dir: str = "queries",
    output_dir: str = "queries/queryresults",
    output_name: str | None = None,
) -> None:
    conn = gcamreader.LocalDBConn(dbpath=dbpath, dbfile=dbfile)
    selected_scenario, default_output_name = resolve_query_scenario(conn, scenario)
    scenario_output_name = output_name or scenario or default_output_name

    query_files = [f for f in os.listdir(query_dir) if f.endswith(".xml")]
    scenario_output_dir = Path(output_dir) / scenario_output_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Database file: {dbfile}")
    print(f"Running GCAM scenario: {selected_scenario}")
    print(f"Writing query results to: {scenario_output_dir}")

    for query_file in query_files:
        queries = gcamreader.parse_batch_query(os.path.join(query_dir, query_file))

        for query in queries:
            print(f"Running query: {query.title}")
            data = conn.runQuery(query, scenarios=[selected_scenario])
            data.to_csv(scenario_output_dir / f"{query.title}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all GCAM batch query XML files and export CSV query results "
            "for a specific scenario inside a GCAM database."
        )
    )
    parser.add_argument(
        "dbfile",
        nargs="?",
        default="SSP2",
        help="GCAM database file name to open (default: SSP2).",
    )
    parser.add_argument(
        "--scenario",
        help=(
            "Scenario name inside the database to query. Accepts either the "
            "scenario name if it is unique or the fully qualified name "
            "'<name> <date>'. Defaults to the most recent scenario in the database."
        ),
    )
    parser.add_argument(
        "--dbpath",
        default="database",
        help="Path to the GCAM database directory (default: database).",
    )
    parser.add_argument(
        "--query-dir",
        default="queries",
        help="Directory containing GCAM query XML files (default: queries).",
    )
    parser.add_argument(
        "--output-dir",
        default="queries/queryresults",
        help="Directory where query CSV files are written (default: queries/queryresults).",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Directory name to use under the output directory. Defaults to the "
            "requested scenario name, or the resolved scenario name if omitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_queries(
        dbfile=args.dbfile,
        scenario=args.scenario,
        dbpath=args.dbpath,
        query_dir=args.query_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
