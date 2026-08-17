from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class MatrixRun:
    dataset: str
    model: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.dataset}__{self.model}__seed{self.seed}__confirmatory"


def frozen_confirmatory_matrix() -> tuple[MatrixRun, ...]:
    """Return the preregistered Tier-B matrix in a stable launch order."""

    models = ("sabet_hard", "tempo_qr_hard")
    primary_datasets = (
        "wikidata_big",
        "wikidata_big_complex",
        "MultiTQ",
        "timequestions",
    )
    runs = [
        MatrixRun(dataset, model, 1729)
        for dataset in primary_datasets
        for model in models
    ]
    runs.extend(
        MatrixRun(dataset, model, seed)
        for dataset in ("wikidata_big_complex", "timequestions")
        for seed in (2718, 3141)
        for model in models
    )
    return tuple(runs)


def run_matrix(
    *,
    runner_path: Path,
    source_root: Path,
    data_root: Path,
    runs_root: Path,
    max_parallel: int = 3,
    poll_seconds: float = 5.0,
) -> dict[str, object]:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    runner_path = runner_path.resolve()
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    runs_root = runs_root.resolve()
    for path, label in (
        (runner_path, "runner"),
        (source_root, "instrumented source"),
        (data_root, "released data"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    runs_root.mkdir(parents=True, exist_ok=True)
    log_root = runs_root / "_matrix_logs"
    log_root.mkdir(exist_ok=True)
    matrix_path = runs_root / "confirmatory_matrix_manifest.json"
    specs = frozen_confirmatory_matrix()
    completed: list[str] = []
    pending: list[MatrixRun] = []
    for spec in specs:
        run_dir = runs_root / spec.run_id
        if not run_dir.exists():
            pending.append(spec)
            continue
        status = _manifest_status(run_dir)
        if status != "completed":
            raise RuntimeError(
                f"Existing run {spec.run_id} has status {status!r}; "
                "the matrix runner will not overwrite or silently resume it"
            )
        _verify_existing(runner_path, run_dir, log_root)
        completed.append(spec.run_id)

    started = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "started_utc": started.isoformat(),
        "matrix_runner_sha256": _sha256(Path(__file__)),
        "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
        "source_root": str(source_root),
        "data_root": str(data_root),
        "runs_root": str(runs_root),
        "max_parallel": max_parallel,
        "poll_seconds": poll_seconds,
        "frozen_runs": [asdict(spec) | {"run_id": spec.run_id} for spec in specs],
        "preexisting_verified_runs": completed.copy(),
        "launched_runs": [],
        "completed_runs": completed.copy(),
    }
    _write_json(matrix_path, manifest)
    active: dict[str, tuple[subprocess.Popen[str], object]] = {}
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                spec = pending.pop(0)
                log_path = log_root / f"{spec.run_id}.launcher.log"
                log_stream = log_path.open("w", encoding="utf-8", newline="\n")
                command = _run_command(
                    runner_path=runner_path,
                    source_root=source_root,
                    data_root=data_root,
                    runs_root=runs_root,
                    spec=spec,
                )
                process = subprocess.Popen(
                    command,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[spec.run_id] = (process, log_stream)
                launched = manifest["launched_runs"]
                assert isinstance(launched, list)
                launched.append(
                    {
                        "run_id": spec.run_id,
                        "pid": process.pid,
                        "launched_utc": datetime.now(UTC).isoformat(),
                        "log_path": str(log_path),
                    }
                )
                _write_json(matrix_path, manifest)

            finished = [
                run_id
                for run_id, (process, _) in active.items()
                if process.poll() is not None
            ]
            if not finished:
                time.sleep(poll_seconds)
                continue
            for run_id in finished:
                process, log_stream = active.pop(run_id)
                log_stream.close()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Run {run_id} failed with return code {process.returncode}; "
                        "remaining matrix runs were not launched"
                    )
                run_dir = runs_root / run_id
                if _manifest_status(run_dir) != "completed":
                    raise RuntimeError(
                        f"Run {run_id} exited successfully without a completed manifest"
                    )
                _verify_existing(runner_path, run_dir, log_root)
                completed.append(run_id)
                completed_runs = manifest["completed_runs"]
                assert isinstance(completed_runs, list)
                completed_runs.append(run_id)
                _write_json(matrix_path, manifest)
    except BaseException as error:
        survivor_results: dict[str, int] = {}
        for process, log_stream in active.values():
            process.wait()
            log_stream.close()
        for run_id, (process, _log_stream) in active.items():
            survivor_results[run_id] = int(process.returncode)
        manifest.update(
            {
                "status": "failed",
                "finished_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "already_launched_sibling_return_codes": survivor_results,
            }
        )
        _write_json(matrix_path, manifest)
        raise

    manifest.update(
        {
            "status": "completed",
            "finished_utc": datetime.now(UTC).isoformat(),
            "completed_run_count": len(completed),
        }
    )
    _write_json(matrix_path, manifest)
    return manifest


def _run_command(
    *,
    runner_path: Path,
    source_root: Path,
    data_root: Path,
    runs_root: Path,
    spec: MatrixRun,
) -> list[str]:
    return [
        sys.executable,
        str(runner_path),
        "--source-root",
        str(source_root),
        "--data-root",
        str(data_root),
        "--runs-root",
        str(runs_root),
        "--dataset",
        spec.dataset,
        "--model",
        spec.model,
        "--seed",
        str(spec.seed),
        "--purpose",
        "confirmatory",
    ]


def _verify_existing(runner_path: Path, run_dir: Path, log_root: Path) -> None:
    log_path = log_root / f"{run_dir.name}.verification.log"
    completed = subprocess.run(
        [sys.executable, str(runner_path), "--verify-existing-run", str(run_dir)],
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
        raise RuntimeError(f"Independent verification failed for {run_dir.name}")


def _manifest_status(run_dir: Path) -> str | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = value.get("status")
    return status if isinstance(status, str) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen SABET confirmatory matrix")
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    result = run_matrix(
        runner_path=args.runner_path,
        source_root=args.source_root,
        data_root=args.data_root,
        runs_root=args.runs_root,
        max_parallel=args.max_parallel,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
