import argparse
import os
from pathlib import Path

import gcamreader


def run_queries(
    scenario: str,
    dbpath: str = "database",
    query_dir: str = "queries",
    output_dir: str = "queries/queryresults",
) -> None:
    conn = gcamreader.LocalDBConn(dbpath=dbpath, dbfile=scenario)

    query_files = [f for f in os.listdir(query_dir) if f.endswith(".xml")]
    scenario_output_dir = Path(output_dir) / scenario
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    for query_file in query_files:
        queries = gcamreader.parse_batch_query(os.path.join(query_dir, query_file))

        for query in queries:
            print(f"Running query: {query.title}")
            data = conn.runQuery(query)
            data.to_csv(scenario_output_dir / f"{query.title}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all GCAM batch query XML files and export CSV query results."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="SSP2",
        help="GCAM database/scenario name (default: SSP2).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_queries(
        scenario=args.scenario,
        dbpath=args.dbpath,
        query_dir=args.query_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
