"""
Memory usage analyzer for TT-NN Neural Network model.

Features:
- Per-operation total L1/DRAM usage (sum over bank/core)
- If an operation has no memory report, reuse the previous op's usage
- Min / Max / Average L1/DRAM usage
- Optional CSV export via --csv
- Optional skip of per-op table via --no-table

Usage:
    python report_memory_occupation.py
    python report_memory_occupation.py --profiler-path /path/to/report
    python report_memory_occupation.py --csv mem_usage.csv
    python report_memory_occupation.py --no-table
    # If --profiler-path is omitted, an interactive prompt lists reports under generated/ttnn/reports
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


REQUIRED_REPORT_FILES = (
    "config.json",
    "cluster_descriptor.yaml",
    "db.sqlite",
)


def format_bytes_mib(num_bytes: int) -> str:
    mib = num_bytes / (1024**2)
    return f"{mib:8.2f} MiB"


def format_stack_trace(stack_trace: Optional[str]) -> str:
    if not stack_trace:
        return "-"
    for seg in str(stack_trace).splitlines():
        stripped = seg.strip()
        if not stripped:
            continue
        cleaned = stripped
        if cleaned.startswith("File "):
            cleaned = cleaned[len("File ") :]
        line_idx = cleaned.find("line ")
        if line_idx != -1:
            prefix = cleaned[: line_idx + len("line ")]
            suffix = cleaned[line_idx + len("line ") :]
            digits = []
            for ch in suffix:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            cleaned = (prefix + "".join(digits)).rstrip()
        return cleaned or "-"
    return "-"


def load_report_metadata(config_path: Path) -> Tuple[str, str]:
    """Return (report_name, report_path) parsed from config.json."""
    try:
        with config_path.open("r", encoding="utf-8") as cfg:
            data = json.load(cfg)
        report_name = data.get("report_name") or "-"
        report_path = data.get("report_path") or "-"
        return str(report_name), str(report_path)
    except (OSError, json.JSONDecodeError):
        return "-", "-"


def gather_profiler_reports(base_dir: Path) -> List[Tuple[Path, str, str]]:
    """Collect (directory, report_name, report_path) tuples for valid reports."""
    reports: List[Tuple[Path, str, str]] = []
    if not base_dir.is_dir():
        return reports

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        required_paths = [child / name for name in REQUIRED_REPORT_FILES]
        if not all(path.is_file() for path in required_paths):
            continue
        report_name, report_path = load_report_metadata(child / "config.json")
        reports.append((child, report_name, report_path))

    return reports


def resolve_profiler_path(explicit_path: Optional[str], base_dir: Path) -> Path:
    if explicit_path:
        profiler_path = Path(explicit_path).expanduser()
        missing = [name for name in REQUIRED_REPORT_FILES if not (profiler_path / name).is_file()]
        if not profiler_path.is_dir() or missing:
            print(
                "ERROR: --profiler-path must point to a directory containing config.json, cluster_descriptor.yaml, and db.sqlite."
            )
            sys.exit(1)
        return profiler_path

    reports = gather_profiler_reports(base_dir)
    if not reports:
        print(f"ERROR: No valid profiler reports found under '{base_dir}'.")
        sys.exit(1)

    print("\nAvailable profiler reports:")
    table_rows = [
        [str(idx), report_path, report_name] for idx, (_, report_name, report_path) in enumerate(reports, start=1)
    ]
    print_table(["No", "Path", "Name"], table_rows)

    while True:
        choice = input("\nEnter report number: ").strip()
        try:
            selection = int(choice)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if 1 <= selection <= len(reports):
            return reports[selection - 1][0]
        print("Select a number from the list.")


def print_table(headers, rows):
    if not rows:
        print("(no rows)")
        return

    col_widths = [len(h) for h in headers]

    # Compute column widths
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Header
    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep)

    # Rows
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))


def forward_fill_usage(series_bytes: pd.Series, has_measure: pd.Series) -> pd.Series:
    # Set values to NaN where has_measure is False, then forward-fill and cast to int64
    filled = series_bytes.where(has_measure).ffill().astype("int64")
    return filled


def analyze(memory_report_path: str, csv_path: Optional[str] = None, print_per_op: bool = True) -> None:
    db_path = os.path.join(memory_report_path, "db.sqlite")
    conn = sqlite3.connect(db_path)

    # Speed up SQLite for read-only analytics
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    # Create an index to speed up GROUP BY on large buffers table (idempotent)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buffers_opid_type ON buffers(operation_id, buffer_type);")

    # Load operations into a DataFrame.
    operations_columns = {row[1] for row in conn.execute("PRAGMA table_info(operations)").fetchall()}
    has_stack_trace_col = "stack_trace" in operations_columns
    has_stack_trace_table = (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stack_traces'").fetchone() is not None
    )

    if has_stack_trace_col:
        select_sql = "SELECT operation_id, name AS op_name, stack_trace FROM operations ORDER BY operation_id"
        stack_trace_source = "column"
    elif has_stack_trace_table:
        select_sql = """
            SELECT o.operation_id,
                   o.name AS op_name,
                   st.stack_trace
            FROM operations o
            LEFT JOIN stack_traces st
              ON o.operation_id = st.operation_id
            ORDER BY o.operation_id
        """
        stack_trace_source = "table"
    else:
        select_sql = "SELECT operation_id, name AS op_name FROM operations ORDER BY operation_id"
        stack_trace_source = "none"

    df_ops = pd.read_sql_query(select_sql, conn)
    if df_ops.empty:
        print("ERROR: operations table is empty.")
        conn.close()
        return
    if stack_trace_source == "none":
        df_ops["stack_trace"] = ""
    elif stack_trace_source == "table":
        df_ops["stack_trace"] = df_ops["stack_trace"].fillna("")

    # Aggregate buffers: total bytes per (operation_id, buffer_type)
    # buffer_type: 0 = DRAM, 1 = L1
    df_usage = pd.read_sql_query(
        """
        SELECT operation_id,
               buffer_type,
               SUM(max_size_per_bank) AS total_bytes
        FROM buffers
        GROUP BY operation_id, buffer_type
        """,
        conn,
    )

    conn.close()

    # Pivot buffer_type into columns: dram_bytes, l1_bytes
    if df_usage.empty:
        # No buffers, just create zero columns keyed by operation_id
        df_usage_pivot = pd.DataFrame(
            0,
            index=df_ops["operation_id"],
            columns=["dram_bytes", "l1_bytes"],
        )
        df_usage_pivot.index.name = "operation_id"
    else:
        df_usage_pivot = df_usage.pivot(index="operation_id", columns="buffer_type", values="total_bytes").rename(
            columns={0: "dram_bytes", 1: "l1_bytes"}
        )

    # Merge operations and usage (without filling NaN yet)
    df = df_ops.merge(df_usage_pivot, on="operation_id", how="left")

    # Determine which ops actually have a memory measurement
    df["has_measure"] = ~(df["l1_bytes"].isna() & df["dram_bytes"].isna())

    # Replace NaN with 0 in raw reported bytes
    df["l1_bytes"] = df["l1_bytes"].fillna(0).astype("int64")
    df["dram_bytes"] = df["dram_bytes"].fillna(0).astype("int64")

    # Forward-fill effective usage when there is no measurement
    df["l1_bytes_eff"] = forward_fill_usage(df["l1_bytes"], df["has_measure"])
    df["dram_bytes_eff"] = forward_fill_usage(df["dram_bytes"], df["has_measure"])

    # Precompute MiB columns using effective values
    df["l1_mib"] = df["l1_bytes_eff"] / (1024**2)
    df["dram_mib"] = df["dram_bytes_eff"] / (1024**2)

    # Per-operation table (optional)
    if print_per_op:
        print("\n=== Per-Operation Memory Usage (effective; sum over bank/core) ===\n")
        print_table(
            ["Op ID", "Operation Name", "L1 Usage", "DRAM Usage", "Stack Trace"],
            [
                [
                    int(row.operation_id),
                    row.op_name,
                    format_bytes_mib(int(row.l1_bytes_eff)),
                    format_bytes_mib(int(row.dram_bytes_eff)),
                    format_stack_trace(row.stack_trace),
                ]
                for row in df.itertuples(index=False)
            ],
        )

    # L1 summary (effective; sum over all banks/cores)
    l1_min_idx = df["l1_bytes_eff"].idxmin()
    l1_max_idx = df["l1_bytes_eff"].idxmax()
    l1_avg = df["l1_bytes_eff"].mean()

    min_l1_row = df.loc[l1_min_idx]
    max_l1_row = df.loc[l1_max_idx]

    # DRAM summary (effective; sum over all banks/cores)
    dram_min_idx = df["dram_bytes_eff"].idxmin()
    dram_max_idx = df["dram_bytes_eff"].idxmax()
    dram_avg = df["dram_bytes_eff"].mean()

    min_dram_row = df.loc[dram_min_idx]
    max_dram_row = df.loc[dram_max_idx]

    print("\n=== L1 Summary (effective; sum over bank/core) ===\n")
    l1_summary_rows = [
        [
            "Min",
            int(min_l1_row.operation_id),
            min_l1_row.op_name,
            format_bytes_mib(int(min_l1_row.l1_bytes_eff)),
        ],
        [
            "Max",
            int(max_l1_row.operation_id),
            max_l1_row.op_name,
            format_bytes_mib(int(max_l1_row.l1_bytes_eff)),
        ],
        [
            "Avg",
            "-",
            "-",
            format_bytes_mib(int(l1_avg)),
        ],
    ]
    print_table(["Type", "Op ID", "Operation Name", "Usage"], l1_summary_rows)

    print("\n=== DRAM Summary (effective; sum over bank/core) ===\n")
    dram_summary_rows = [
        [
            "Min",
            int(min_dram_row.operation_id),
            min_dram_row.op_name,
            format_bytes_mib(int(min_dram_row.dram_bytes_eff)),
        ],
        [
            "Max",
            int(max_dram_row.operation_id),
            max_dram_row.op_name,
            format_bytes_mib(int(max_dram_row.dram_bytes_eff)),
        ],
        [
            "Avg",
            "-",
            "-",
            format_bytes_mib(int(dram_avg)),
        ],
    ]
    print_table(["Type", "Op ID", "Operation Name", "Usage"], dram_summary_rows)

    # CSV export (effective usage)
    if csv_path:
        df_out = df[
            [
                "operation_id",
                "op_name",
                "l1_mib",
                "dram_mib",
                "stack_trace",
            ]
        ].copy()
        df_out["stack_trace"] = df_out["stack_trace"].apply(format_stack_trace)
        df_out = df_out.rename(
            columns={
                "operation_id": "Op ID",
                "op_name": "Operation Name",
                "l1_mib": "L1 Usage (MiB)",
                "dram_mib": "DRAM Usage (MiB)",
                "stack_trace": "Stack Trace",
            },
        )
        df_out.to_csv(csv_path, index=False)
        print(f"\n[INFO] CSV saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool for analyzing TT-NN Neural Network model L1/DRAM memory usage."
    )
    parser.add_argument(
        "--profiler-path",
        help=("Specify a profiler path"),
    )
    parser.add_argument(
        "--csv",
        metavar="CSV_PATH",
        help="If set, save per-operation DRAM/L1 usage to CSV file (effective values).",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="If set, skip printing the per-operation table (faster for large traces).",
    )
    args = parser.parse_args()

    analyze(
        str(
            resolve_profiler_path(
                args.profiler_path,
                Path("generated/ttnn/reports").expanduser(),
            )
        ),
        csv_path=args.csv,
        print_per_op=not args.no_table,
    )
