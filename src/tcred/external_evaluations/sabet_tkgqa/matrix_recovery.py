from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.matrix_runner import (
    MatrixRun,
    _run_command,
    _verify_existing,
    frozen_confirmatory_matrix,
)
from tcred.external_evaluations.sabet_tkgqa.runner import DATASETS, _sha256, _write_json


@dataclass
class ActiveRun:
    spec: MatrixRun
    process: subprocess.Popen[str] | None
    log_stream: object | None
    adopted: bool
    observed_monotonic: float


def recover_matrix(
    *,
    experiment_python: Path,
    runner_path: Path,
    label_correction_path: Path,
    source_root: Path,
    data_root: Path,
    runs_root: Path,
    manifest_path: Path,
    adopt_running: set[str],
    predecessor_manifest_path: Path | None = None,
    max_parallel: int = 3,
    poll_seconds: float = 5.0,
    startup_timeout_seconds: float = 5 * 60,
    adopted_timeout_seconds: float = 6 * 60 * 60,
) -> dict[str, object]:
    """Continue a frozen matrix while explicitly adopting audited in-progress runs."""

    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if startup_timeout_seconds <= 0:
        raise ValueError("startup_timeout_seconds must be positive")
    if adopted_timeout_seconds <= 0:
        raise ValueError("adopted_timeout_seconds must be positive")
    experiment_python = experiment_python.resolve()
    runner_path = runner_path.resolve()
    label_correction_path = label_correction_path.resolve()
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    runs_root = runs_root.resolve()
    manifest_path = manifest_path.resolve()
    predecessor_manifest_path = (
        predecessor_manifest_path.resolve()
        if predecessor_manifest_path is not None
        else None
    )
    for path, label in (
        (experiment_python, "experiment Python interpreter"),
        (runner_path, "runner"),
        (label_correction_path, "label-correction utility"),
        (source_root, "instrumented source"),
        (data_root, "released data"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery manifest: {manifest_path}")
    if predecessor_manifest_path is not None and not predecessor_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing predecessor matrix manifest: {predecessor_manifest_path}"
        )

    specs = frozen_confirmatory_matrix()
    known_run_ids = {spec.run_id for spec in specs}
    unknown_adoptions = sorted(adopt_running - known_run_ids)
    if unknown_adoptions:
        raise ValueError(f"Unknown adopted run IDs: {unknown_adoptions}")

    log_root = runs_root / "_matrix_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    pending: list[MatrixRun] = []
    active: dict[str, ActiveRun] = {}
    adopted_completed_before_start: list[str] = []
    corrected_runs: list[dict[str, object]] = []
    for spec in specs:
        run_dir = runs_root / spec.run_id
        if not run_dir.exists():
            if spec.run_id in adopt_running:
                raise FileNotFoundError(f"Adopted run directory is missing: {run_dir}")
            pending.append(spec)
            continue
        run_manifest = _read_run_manifest(run_dir)
        status = run_manifest.get("status")
        if status == "completed":
            _verify_existing(runner_path, run_dir, log_root)
            completed.append(spec.run_id)
            if spec.run_id in adopt_running:
                adopted_completed_before_start.append(spec.run_id)
            continue
        if status == "invalid_outputs" and spec.run_id in adopt_running:
            correction = _correct_invalid_output(
                label_correction_path=label_correction_path,
                run_dir=run_dir,
                log_root=log_root,
            )
            corrected_runs.append(correction)
            _verify_existing(runner_path, run_dir, log_root)
            completed.append(spec.run_id)
            adopted_completed_before_start.append(spec.run_id)
            continue
        if status not in {"running", "process_completed"}:
            raise RuntimeError(
                f"Existing run {spec.run_id} has nonrecoverable status {status!r}"
            )
        if spec.run_id not in adopt_running:
            raise RuntimeError(
                f"Existing in-progress run requires explicit adoption: {spec.run_id}"
            )
        _validate_adopted_manifest(run_manifest, spec)
        active[spec.run_id] = ActiveRun(
            spec=spec,
            process=None,
            log_stream=None,
            adopted=True,
            observed_monotonic=time.monotonic(),
        )

    unresolved_adoptions = adopt_running - set(active) - set(adopted_completed_before_start)
    if unresolved_adoptions:
        raise RuntimeError(f"Requested adoptions were not resolved: {sorted(unresolved_adoptions)}")
    if len(active) > max_parallel:
        raise RuntimeError(
            f"Adopted {len(active)} active runs, exceeding max_parallel={max_parallel}"
        )

    started = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "purpose": "fail-closed continuation of the frozen confirmatory matrix",
        "started_utc": started.isoformat(),
        "recovery_implementation_sha256": _sha256(Path(__file__)),
        "experiment_python": str(experiment_python),
        "recovery_python": sys.executable,
        "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
        "label_correction_utility": {
            "path": str(label_correction_path),
            "sha256": _sha256(label_correction_path),
        },
        "source_root_for_new_runs": str(source_root),
        "data_root": str(data_root),
        "runs_root": str(runs_root),
        "manifest_path": str(manifest_path),
        "predecessor_manifest_at_start": (
            _file_identity(predecessor_manifest_path)
            if predecessor_manifest_path is not None
            else None
        ),
        "max_parallel": max_parallel,
        "poll_seconds": poll_seconds,
        "startup_timeout_seconds": startup_timeout_seconds,
        "adopted_timeout_seconds": adopted_timeout_seconds,
        "frozen_runs": [
            {"dataset": spec.dataset, "model": spec.model, "seed": spec.seed, "run_id": spec.run_id}
            for spec in specs
        ],
        "explicitly_adopted_run_ids": sorted(adopt_running),
        "adopted_active_at_start": sorted(active),
        "adopted_completed_before_start": sorted(adopted_completed_before_start),
        "preexisting_verified_runs": completed.copy(),
        "launched_runs": [],
        "corrected_runs": corrected_runs,
        "completed_runs": completed.copy(),
    }
    _write_json(manifest_path, manifest)

    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                spec = pending.pop(0)
                log_path = log_root / f"{spec.run_id}.recovery-v5.launcher.log"
                log_stream = log_path.open("x", encoding="utf-8", newline="\n")
                command = _run_command(
                    runner_path=runner_path,
                    source_root=source_root,
                    data_root=data_root,
                    runs_root=runs_root,
                    spec=spec,
                )
                command[0] = str(experiment_python)
                process = subprocess.Popen(
                    command,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[spec.run_id] = ActiveRun(
                    spec=spec,
                    process=process,
                    log_stream=log_stream,
                    adopted=False,
                    observed_monotonic=time.monotonic(),
                )
                launched = manifest["launched_runs"]
                assert isinstance(launched, list)
                launched.append(
                    {
                        "run_id": spec.run_id,
                        "pid": process.pid,
                        "launched_utc": datetime.now(UTC).isoformat(),
                        "command": command,
                        "log_path": str(log_path),
                    }
                )
                _write_json(manifest_path, manifest)

            finished: list[str] = []
            for run_id, state in active.items():
                run_dir = runs_root / run_id
                try:
                    run_manifest = _read_run_manifest(run_dir)
                except RuntimeError:
                    return_code = (
                        state.process.poll() if state.process is not None else None
                    )
                    startup_elapsed = time.monotonic() - state.observed_monotonic
                    if (
                        not state.adopted
                        and return_code is None
                        and startup_elapsed <= startup_timeout_seconds
                    ):
                        continue
                    raise
                status = run_manifest.get("status")
                if status == "completed":
                    if state.process is not None:
                        return_code = state.process.poll()
                        if return_code is None:
                            continue
                        if return_code != 0:
                            raise RuntimeError(
                                f"Run {run_id} completed its manifest but exited {return_code}"
                            )
                    _verify_existing(runner_path, run_dir, log_root)
                    finished.append(run_id)
                    continue
                if status == "invalid_outputs":
                    if state.process is not None and state.process.poll() is None:
                        continue
                    correction = _correct_invalid_output(
                        label_correction_path=label_correction_path,
                        run_dir=run_dir,
                        log_root=log_root,
                    )
                    corrected_runs.append(correction)
                    _verify_existing(runner_path, run_dir, log_root)
                    finished.append(run_id)
                    continue
                if status == "failed":
                    raise RuntimeError(f"Run {run_id} reported failed status")
                if status not in {"running", "process_completed"}:
                    raise RuntimeError(f"Run {run_id} reported unexpected status {status!r}")
                if state.process is not None:
                    return_code = state.process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            f"Run {run_id} exited {return_code} with status {status!r}"
                        )
                elif time.monotonic() - state.observed_monotonic > adopted_timeout_seconds:
                    raise TimeoutError(f"Adopted run did not finish before timeout: {run_id}")

            if not finished:
                time.sleep(poll_seconds)
                continue
            for run_id in finished:
                state = active.pop(run_id)
                if state.log_stream is not None:
                    state.log_stream.close()
                completed.append(run_id)
                completed_runs = manifest["completed_runs"]
                assert isinstance(completed_runs, list)
                completed_runs.append(run_id)
            manifest["corrected_runs"] = corrected_runs
            _write_json(manifest_path, manifest)
    except BaseException as error:
        owned_return_codes: dict[str, int] = {}
        for run_id, state in active.items():
            if state.process is None:
                continue
            state.process.wait()
            owned_return_codes[run_id] = int(state.process.returncode)
            if state.log_stream is not None:
                state.log_stream.close()
        manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "owned_sibling_return_codes": owned_return_codes,
                "corrected_runs": corrected_runs,
            }
        )
        _write_json(manifest_path, manifest)
        raise

    manifest.update(
        {
            "status": "completed",
            "finished_utc": datetime.now(UTC).isoformat(),
            "completed_run_count": len(completed),
            "corrected_runs": corrected_runs,
            "predecessor_manifest_at_finish": (
                _file_identity(predecessor_manifest_path)
                if predecessor_manifest_path is not None
                else None
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def _validate_adopted_manifest(manifest: dict[str, object], spec: MatrixRun) -> None:
    expected = {
        "run_id": spec.run_id,
        "purpose": "confirmatory",
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "epochs": DATASETS[spec.dataset].epochs,
        "epochs_overridden": False,
        "deterministic_algorithms": True,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"Adopted run contract mismatch for {spec.run_id}: "
                f"{name}={manifest.get(name)!r}, expected {value!r}"
            )


def _correct_invalid_output(
    *, label_correction_path: Path, run_dir: Path, log_root: Path
) -> dict[str, object]:
    log_path = log_root / f"{run_dir.name}.label-correction.log"
    completed = subprocess.run(
        [sys.executable, str(label_correction_path), "--run-dir", str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Audited label correction failed for {run_dir.name}")
    manifest = _read_run_manifest(run_dir)
    correction = manifest.get("label_correction")
    if manifest.get("status") != "completed" or not isinstance(correction, dict):
        raise RuntimeError(f"Correction did not complete run {run_dir.name}")
    return {
        "run_id": run_dir.name,
        "corrected_utc": correction.get("corrected_utc"),
        "replacement_count": correction.get("replacement_count"),
        "changed_row_count": correction.get("changed_row_count"),
        "original_sha256": correction.get("original_sha256"),
        "corrected_sha256": correction.get("corrected_sha256"),
        "log_path": str(log_path),
    }


def _read_run_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_manifest.json"
    last_error: Exception | None = None
    for _ in range(5):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Run manifest is not an object: {path}")
            return value
        except (FileNotFoundError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.2)
    raise RuntimeError(f"Cannot read a complete run manifest: {path}") from last_error


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue the frozen SABET matrix with explicit in-progress adoption"
    )
    parser.add_argument("--experiment-python", type=Path, required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--label-correction-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--predecessor-manifest", type=Path)
    parser.add_argument("--adopt-running", nargs="*", default=[])
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=5 * 60)
    parser.add_argument("--adopted-timeout-seconds", type=float, default=6 * 60 * 60)
    args = parser.parse_args()
    result = recover_matrix(
        experiment_python=args.experiment_python,
        runner_path=args.runner_path,
        label_correction_path=args.label_correction_path,
        source_root=args.source_root,
        data_root=args.data_root,
        runs_root=args.runs_root,
        manifest_path=args.manifest_path,
        adopt_running=set(args.adopt_running),
        predecessor_manifest_path=args.predecessor_manifest,
        max_parallel=args.max_parallel,
        poll_seconds=args.poll_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        adopted_timeout_seconds=args.adopted_timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
