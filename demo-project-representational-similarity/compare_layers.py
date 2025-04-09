import mlflow
import torch
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, GraphModulePlus
from nn_lib.utils import search_runs_by_params
from torch.utils.data import Subset, DataLoader

from comparators import Comparator


def _get_test_data(dataset_name: str, seed: int, m: int):
    rng = torch.Generator()
    rng.manual_seed(seed)
    if dataset_name == "imagenet":
        dm = ImageNetDataModule(root_dir="/data/datasets", seed=seed)
        dm.prepare_data()
        dm.setup("test")
        dataset = Subset(
            dm.test_ds, indices=torch.randperm(len(dm.test_ds), generator=rng)[:m]
        )
        return DataLoader(
            dataset, batch_size=100, num_workers=4, shuffle=False, pin_memory=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_compare(
    modelA: str,
    layerA: str,
    modelB: str,
    layerB: str,
    dataset: str,
    comparator: Comparator,
    m: int = 1000,
    data_seed: int = 189645,
    device: str | torch.device = "cuda",
):
    getterA = (
        GraphModulePlus.new_from_trace(get_pretrained_model(modelA))
        .set_output(layerA)
        .eval()
        .to(device)
    )
    getterB = (
        GraphModulePlus.new_from_trace(get_pretrained_model(modelB))
        .set_output(layerB)
        .eval()
        .to(device)
    )

    data = _get_test_data(dataset, data_seed, m)

    repsA, repsB = [], []
    with torch.no_grad():
        for im, _ in data:
            im = im.to(device)
            repsA.append(getterA(im))
            repsB.append(getterB(im))

    score = comparator.compare(
        torch.cat(repsA, dim=0),
        torch.cat(repsB, dim=0),
    )

    mlflow.log_metric("score", score.item())


def _flatten_dict(d: dict, key_sep="_") -> dict:
    """Flattens a nested dictionary."""
    out = {}

    def flatten(x: dict, name: str = ""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + key_sep)
        else:
            if name[:-1] in out:
                raise ValueError(
                    f"Duplicate key created during flattening: {name[:-1]}"
                )
            out[name[:-1]] = x

    flatten(d)
    return out


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(run_compare)
    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri("/data/projects/demo-similarity/mlruns")
    mlflow.set_experiment("demo-similarity")

    # Check if a run with the same parameters already exists, and remove 'force_recompute' from args
    # to avoid it being considered as a parameter below
    existing_runs = search_runs_by_params(
        "demo-similarity",
        finished_only=True,
        params=_flatten_dict(args.as_dict()),
        skip_fields={"device": ...},
    )
    if not existing_runs.empty:
        print("Run already exists with these params. Skipping.")
        exit(0)

    # Start the run and log params
    args_instantiated = parser.instantiate_classes(args)
    with mlflow.start_run() as run:
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_compare(**args_instantiated.as_dict())
