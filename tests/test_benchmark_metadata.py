from __future__ import annotations

import json

import pytest

from bench.report import validate_run
from bench.workloads import BENCHMARK_SUITE, P0_MODELS, WORKLOADS


def complete_run() -> dict:
    return {
        "benchmark_suite": BENCHMARK_SUITE,
        "models": {
            model_key: {
                "workloads": {
                    workload.name: {"mean_ms": 1.0}
                    for workload in WORKLOADS
                    if workload.kind in spec["roles"]
                }
            }
            for model_key, spec in P0_MODELS.items()
        },
    }


def test_validate_run_accepts_current_suite(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(complete_run()))

    validate_run(path, complete_run())


def test_validate_run_rejects_stale_model_set(tmp_path):
    run = complete_run()
    run["models"].pop(next(iter(run["models"])))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))

    with pytest.raises(ValueError, match="model set"):
        validate_run(path, run)


def test_validate_run_rejects_wrong_suite(tmp_path):
    run = complete_run()
    run["benchmark_suite"] = "old-suite"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))

    with pytest.raises(ValueError, match="benchmark suite"):
        validate_run(path, run)
