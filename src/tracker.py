"""
src/tracker.py
--------------
Lightweight experiment tracker that mirrors the MLflow API.
Logs run metadata to JSON files under mlruns/.

On a machine with MLflow installed, swap the import and calls:
    import mlflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime


class ExperimentTracker:
    def __init__(self, experiment_name: str, base_dir: Path = None):
        self.experiment_name = experiment_name
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "mlruns"
        self.exp_dir = Path(base_dir) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, run_name: str, params: dict, metrics: dict,
                tags: dict = None, artifacts: list = None):
        run_id   = hashlib.md5(f"{run_name}{time.time()}".encode()).hexdigest()[:8]
        filename = f"{run_name}_{run_id}.json"

        record = {
            "run_id":          run_id,
            "run_name":        run_name,
            "experiment":      self.experiment_name,
            "start_time":      datetime.now().isoformat(),
            "status":          "FINISHED",
            "params":          params,
            "metrics":         metrics,
            "tags":            tags or {},
            "artifacts":       artifacts or [],
        }

        path = self.exp_dir / filename
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

        print(f"[mlflow] Run '{run_name}' logged → mlruns/{self.experiment_name}/{filename}")
        return record

    def load_runs(self) -> list:
        runs = []
        for p in sorted(self.exp_dir.glob("*.json")):
            with open(p) as f:
                runs.append(json.load(f))
        return runs
