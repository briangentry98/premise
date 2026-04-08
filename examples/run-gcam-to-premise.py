import argparse
import os
import traceback
from pathlib import Path

import bw2data
import bw2io as bi

from premise import NewDatabase

ROOT = Path(__file__).resolve().parents[1]


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of the IAM files were found: {candidates}")


def ensure_biosphere_database(name: str) -> None:
    if name not in bw2data.databases:
        print(f"Creating missing biosphere database: {name}")
        bi.create_default_biosphere3()


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")

    if not key:
        return None

    return key, value


def load_env_file(path: Path) -> bool:
    if not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        # Keep shell-provided values as highest precedence.
        os.environ.setdefault(key, value)

    return True


def load_credentials_from_dotenv() -> list[Path]:
    candidates = [ROOT / ".env", ROOT / "examples" / ".env"]
    loaded = []
    for candidate in candidates:
        if load_env_file(candidate):
            loaded.append(candidate)
    return loaded


def resolve_iam_file(scenario_name: str, iam_file: str | None) -> Path:
    if iam_file:
        path = Path(iam_file)
        return path if path.is_absolute() else (ROOT / path)

    return first_existing_path(
        [ROOT / "gcam" / "output" / scenario_name / f"gcam_{scenario_name}.xlsx"]
    )


def ensure_source_database(source_db: str, version: str, system_model: str) -> None:
    if source_db in bw2data.databases and len(bw2data.Database(source_db)) > 0:
        print(
            f"Found existing Brightway database: {source_db} "
            f"({len(bw2data.Database(source_db))} datasets)"
        )
        return

    username = os.getenv("ECOINVENT_USERNAME")
    password = os.getenv("ECOINVENT_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing ecoinvent credentials. Set ECOINVENT_USERNAME and "
            "ECOINVENT_PASSWORD environment variables."
        )

    importer = bi.import_ecoinvent_release(
        version=version,
        system_model=system_model,
        username=username,
        password=password,
        use_mp=False,
    )
    importer.apply_strategies()


def run_pipeline(args: argparse.Namespace) -> None:
    loaded_env_files = load_credentials_from_dotenv()
    if loaded_env_files:
        print(
            "Loaded environment values from: "
            + ", ".join(str(path.relative_to(ROOT)) for path in loaded_env_files)
        )

    bw2data.projects.set_current(args.project)
    print(bw2data.databases)

    print("=" * 60)
    print(f"STEP 1: Import and setup {args.source_db} database")
    print("=" * 60)
    ensure_source_database(args.source_db, args.source_version, args.system_model)

    print("=" * 60)
    print("STEP 2: Ensure biosphere database")
    print("=" * 60)
    ensure_biosphere_database(args.biosphere_db)

    print("=" * 60)
    print("STEP 3: Set up and process GCAM scenario")
    print("=" * 60)

    iam_file = resolve_iam_file(args.pathway, args.iam_file)
    print(f"Using IAM file: {iam_file}")

    ndb = NewDatabase(
        scenarios=[
            {
                "model": args.model,
                "pathway": args.pathway,
                "year": args.year,
                "filepath": str(iam_file.parent),
            }
        ],
        source_db=args.source_db,
        source_version=args.source_version,
        biosphere_name=args.biosphere_db,
        keep_source_db_uncertainty=args.keep_source_db_uncertainty,
        keep_imports_uncertainty=args.keep_imports_uncertainty,
    )

    print("NewDatabase created successfully")
    print("Updating all sectors with IAM data...")
    ndb.update()

    print("=" * 60)
    print("STEP 4: Validate and export to Brightway")
    print("=" * 60)

    try:
        print("Writing database to Brightway...")
        ndb.write_db_to_brightway()
        print("SUCCESS: scenario written to Brightway!")
    except Exception as exc:
        print(f"ERROR during write_db_to_brightway(): {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("=" * 60)
    print("Premise run completed.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GCAM IAMC file through premise and export to Brightway."
    )
    parser.add_argument("--project", default="premise", help="Brightway project name.")
    parser.add_argument("--model", default="gcam", help="IAM model name.")
    parser.add_argument("--pathway", default="SSP2", help="Scenario/pathway name.")
    parser.add_argument("--year", type=int, default=2050, help="Target year.")
    parser.add_argument(
        "--iam-file",
        default=None,
        help=(
            "Path to IAMC workbook. If omitted, uses "
            "gcam/output/<pathway>/gcam_<pathway>.xlsx"
        ),
    )
    parser.add_argument(
        "--source-db",
        default="ecoinvent-3.11-cutoff",
        help="Brightway source database name.",
    )
    parser.add_argument(
        "--source-version",
        default="3.11",
        help="Source database version for premise.",
    )
    parser.add_argument(
        "--system-model",
        default="cutoff",
        help="Ecoinvent system model for auto-import.",
    )
    parser.add_argument(
        "--biosphere-db",
        default="ecoinvent-3.11-biosphere",
        help="Biosphere database name.",
    )
    parser.add_argument(
        "--keep-source-db-uncertainty",
        action="store_true",
        help="Keep source database uncertainty in generated database.",
    )
    parser.add_argument(
        "--keep-imports-uncertainty",
        action="store_true",
        help="Keep uncertainty from imported inventories.",
    )
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
