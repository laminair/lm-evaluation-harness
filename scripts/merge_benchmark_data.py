#!/usr/bin/env python3
"""
Merge benchmark data from raw_data folder into a single parquet file.

For each benchmark and document ID, there is one row containing:
- The input text (question)
- For every Qwen model: solved status, energy, latency, throughput, token counts
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, List

import pandas as pd


def extract_timestamp(filename: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)", filename)
    if match:
        return match.group(1)
    return ""


def get_newest_runs(raw_data_dir: Path) -> dict:
    runs = defaultdict(list)
    for model_dir in raw_data_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for benchmark_dir in model_dir.iterdir():
            if not benchmark_dir.is_dir():
                continue
            benchmark_name = benchmark_dir.name
            for sub_dir in benchmark_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                for result_file in sub_dir.glob("results_*.json"):
                    ts = extract_timestamp(result_file.name)
                    runs[(model_name, benchmark_name)].append((ts, result_file))

    newest_runs = {}
    for key, files in runs.items():
        files.sort(key=lambda x: x[0], reverse=True)
        newest_runs[key] = files[0][1]

    return newest_runs


def get_samples_file(results_file: Path, benchmark_name: str) -> Path:
    ts_match = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)", results_file.name
    )
    if ts_match:
        ts = ts_match.group(1)
        samples_name = f"samples_{benchmark_name}_{ts}.jsonl"
        return results_file.parent / samples_name
    return results_file.parent / f"samples_{benchmark_name}.jsonl"


def extract_doc_id(record: dict) -> Optional[str]:
    doc = record.get("doc", {})
    if "id" in doc:
        return str(doc["id"])
    if "idx" in doc:
        return str(doc["idx"])
    if "doc_id" in record:
        return str(record["doc_id"])
    return None


def extract_input_text(doc: dict, benchmark: str) -> Optional[str]:
    if "question" in doc:
        return doc["question"]
    return None


def extract_choices(doc: dict) -> Optional[List]:
    if "choices" in doc:
        return doc["choices"]
    return None


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.replace("Qwen3.5-", "")
    return model_name


def load_single_run(
    results_file: Path, model_name: str, benchmark: str
) -> pd.DataFrame:
    samples_file = get_samples_file(results_file, benchmark)
    if not samples_file.exists():
        print(f"Warning: samples file not found: {samples_file}")
        return pd.DataFrame()

    records = []
    with open(samples_file, "r") as f:
        for line in f:
            record = json.loads(line)

            doc = record.get("doc", {})
            doc_id = extract_doc_id(record)
            if doc_id is None:
                continue

            input_text = extract_input_text(doc, benchmark)
            choices = str(extract_choices(doc)) if extract_choices(doc) else None
            target = record.get("target")

            energy_joules = record.get("energy_joules", float("nan"))
            token_usage = record.get("token_usage", {})
            latency_ms = record.get("latency_ms", float("nan"))
            throughput = record.get("throughput", float("nan"))
            acc = record.get("acc", float("nan"))
            acc_norm = record.get("acc_norm", float("nan"))

            prompt_tokens = (
                token_usage.get("prompt_tokens")
                if isinstance(token_usage, dict)
                else None
            )
            completion_tokens = (
                token_usage.get("completion_tokens")
                if isinstance(token_usage, dict)
                else None
            )
            total_tokens = (
                token_usage.get("total_tokens")
                if isinstance(token_usage, dict)
                else None
            )

            records.append(
                {
                    "doc_id": doc_id,
                    "input_text": input_text,
                    "correct_answer": target,
                    "choices": choices,
                    "model": normalize_model_name(model_name),
                    "solved": acc,
                    "solved_norm": acc_norm,
                    "energy_joules": energy_joules,
                    "latency_ms": latency_ms,
                    "throughput": throughput,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )

    return pd.DataFrame(records)


def main():
    raw_data_dir = Path("raw_data")
    output_file = Path("merged_benchmark_data.parquet")

    print("Finding newest runs for each model/benchmark combination...")
    newest_runs = get_newest_runs(raw_data_dir)
    print(f"Found {len(newest_runs)} runs")

    all_dfs = []
    for (model_name, benchmark_name), results_file in newest_runs.items():
        print(f"Loading {model_name}/{benchmark_name}...")
        df = load_single_run(results_file, model_name, benchmark_name)
        if not df.empty:
            df["benchmark"] = benchmark_name
            all_dfs.append(df)

    if not all_dfs:
        print("No data found!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total records before pivot: {len(combined_df)}")

    id_cols = ["doc_id", "benchmark", "input_text", "correct_answer", "choices"]
    metric_cols = [
        "solved",
        "solved_norm",
        "energy_joules",
        "latency_ms",
        "throughput",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    base_info_df = combined_df[id_cols].copy()
    base_info_df = base_info_df.drop_duplicates(subset=["doc_id", "benchmark"])

    pivot_dfs = []
    for model in combined_df["model"].unique():
        model_df = combined_df[combined_df["model"] == model].copy()
        model_df = model_df.drop(columns=["model"])

        for col in metric_cols:
            pivot_df = model_df.pivot_table(
                index=["doc_id", "benchmark"], values=col, aggfunc="first"
            )
            pivot_df = pivot_df.rename(columns={col: f"{model}_{col}"})
            pivot_dfs.append(pivot_df)

    result_df = pivot_dfs[0]
    for pdf in pivot_dfs[1:]:
        result_df = result_df.join(pdf, how="outer")

    result_df = result_df.reset_index()
    base_df = result_df.merge(base_info_df, on=["doc_id", "benchmark"], how="left")

    column_order = id_cols.copy()
    models = sorted(
        base_df.columns.str.replace(
            r"_(solved|solved_norm|energy_joules|latency_ms|throughput|prompt_tokens|completion_tokens|total_tokens)$",
            "",
            regex=True,
        ).unique()
    )
    for model in models:
        if model in ["doc_id", "benchmark", "input_text", "correct_answer", "choices"]:
            continue
        for suffix in metric_cols:
            col_name = f"{model}_{suffix}"
            if col_name in base_df.columns:
                column_order.append(col_name)

    final_df = base_df[[c for c in column_order if c in base_df.columns]]

    print(f"Final shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")

    print(f"\nSaving to {output_file}...")
    final_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Saved {len(final_df)} rows to {output_file}")

    print("\n--- Summary ---")
    print(
        f"Models: {sorted(final_df.columns.str.replace(r'_(solved|solved_norm|energy_joules|latency_ms|throughput|prompt_tokens|completion_tokens|total_tokens)$', '', regex=True).unique())}"
    )
    print(f"Benchmarks: {sorted(final_df['benchmark'].unique())}")
    print(f"Unique documents: {final_df['doc_id'].nunique()}")


if __name__ == "__main__":
    main()
