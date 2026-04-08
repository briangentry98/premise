import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"Running: {' '.join(command)} (cwd={cwd})")
    subprocess.run(command, cwd=str(cwd), check=True)


def run_pipeline(scenario: str, dbpath: str) -> None:
    gcam_dir = Path(__file__).resolve().parent
    iamc_dir = gcam_dir / "iamc_template"

    run_step(
        [
            sys.executable,
            "GCAM-Query.py",
            scenario,
            "--dbpath",
            dbpath,
        ],
        cwd=gcam_dir,
    )

    run_step(
        [
            sys.executable,
            "run_iamctemplatecreator.py",
            scenario,
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
        "scenario",
        nargs="?",
        default="SSP2",
        help="Scenario/database name to process (default: SSP2).",
    )
    parser.add_argument(
        "--dbpath",
        default="database",
        help="Path to GCAM database directory relative to gcam/ (default: database).",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    run_pipeline(scenario=args.scenario, dbpath=args.dbpath)


if __name__ == "__main__":
    main()
