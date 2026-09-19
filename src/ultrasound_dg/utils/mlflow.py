from __future__ import annotations

import os
from typing import Any

import mlflow
import torch


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    set_experiment: bool = True,
    artifact_root: str | None = None,
) -> None:
    """
    Mlflow configuration:
    - tracking_uri: "http://127.0.0.1:8080"
    - artifact_root: file:./artifacts/mlflow/artifacts
    """
    # Defaults
    experiment_name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "UltrasoundDG"
    )
    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"
    )

    mlflow.set_tracking_uri(tracking_uri)
    if set_experiment:
        mlflow.set_experiment(experiment_name)


def log_config(cfg: Any, name: str) -> None:
    mlflow.log_dict(cfg.model_dump(mode="json"), f"{name}.json")


def load_model_from_mlflow(
    *,
    run_id: str | None = None,
    model_uri: str | None = None,
    model_artifact_path: str | None = "best_model",
    tracking_uri: str | None = None,
    eval_mode: bool = True,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """
    Load a PyTorch model logged in MLflow.

    Exactly one of:
    - run_id  -> loads from runs:/<run_id>/<model_artifact_path>
    - model_uri -> loads from any MLflow URI, e.g. models:/name/1

    """

    if (run_id is None) == (model_uri is None):
        raise ValueError("Provide exactly one of: run_id or model_uri.")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri is None:
        model_uri = f"runs:/{run_id}/{model_artifact_path}"

    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(device))
    model = model.to(device)

    if eval_mode:
        model.eval()

    return model, model_uri
