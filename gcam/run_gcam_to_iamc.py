import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"Running: {' '.join(command)} (cwd={cwd})")
    subprocess.run(command, cwd=str(cwd), check=True)


def run_pipeline(
    dbfile: str,
    dbpath: str,
    scenario: str | None = None,
    output_name: str | None = None,
) -> None:
    gcam_dir = Path(__file__).resolve().parent
    iamc_dir = gcam_dir / "iamc_template"
    scenario_output_name = output_name or scenario or dbfile

    query_command = [
        sys.executable,
        "GCAM-Query.py",
        dbfile,
        "--dbpath",
        dbpath,
    ]
    if scenario:
        query_command.extend(["--scenario", scenario])
    if output_name:
        query_command.extend(["--output-name", output_name])

    run_step(
        query_command,
        cwd=gcam_dir,
    )

    run_step(
        [
            sys.executable,
            "run_iamctemplatecreator.py",
            scenario_output_name,
        ],
        cwd=iamc_dir,
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GCAM queries and IAMC template creation in one command, "
            "producing gcam_<SCENARIO>.xlsx in gcam/output/<SCENARIO>."
        )
    )
    parser.add_argument(
        "dbfile",
        nargs="?",
        default="SSP2",
        help="GCAM database file name to process (default: SSP2).",
    )
    parser.add_argument(
        "--scenario",
        help=(
            "Scenario name inside the database to query. Accepts either the "
            "scenario name if it is unique or the fully qualified name "
            "'<name> <date>'."
        ),
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Directory name to use under gcam/queries/queryresults and gcam/output. "
            "Defaults to the requested scenario name, or the database file name if omitted."
        ),
    )
    parser.add_argument(
        "--dbpath",
        default="database",
        help="Path to GCAM database directory relative to gcam/ (default: database).",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    run_pipeline(
        dbfile=args.dbfile,
        dbpath=args.dbpath,
        scenario=args.scenario,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
