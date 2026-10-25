import itertools
from dataclasses import dataclass
from typing import Optional, Literal

import mlflow
import torch
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, GraphModulePlus
from nn_lib.analysis.similarity import LinearCKA, HSICEstimator
from nn_lib.utils import RunIndex, logged_run
from torch.utils.data import Subset, DataLoader


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


def _get_model(model_name: str):
    return GraphModulePlus.new_from_trace(
        get_pretrained_model(model_name)
    ).squash_all_conv_batchnorm_pairs()


@torch.no_grad()
def _get_reps_iter(
    dl: DataLoader,
    getterA: GraphModulePlus,
    getterB: GraphModulePlus,
    device: str | torch.device,
):
    for batch, _ in dl:
        batch = batch.to(device, non_blocking=True)
        yield getterA(batch), getterB(batch)


@dataclass
class ComparatorSpec:
    method: Literal["cka"] = "cka"  # to be extended another time
    estimator: HSICEstimator = HSICEstimator.CHUN2025

    def get_comparator(self):
        if self.method == "cka":
            return LinearCKA(estimator=self.estimator)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def main(
    modelA: str,
    modelB: str,
    dataset: str,
    comparator: ComparatorSpec,
    m: int = 1000,
    data_seed: int = 189645,
    device: str | torch.device = "cuda",
    layerA: Optional[str] = None,
    layerB: Optional[str] = None,
):
    if layerA is None:
        layersA = [
            str(node) for node in _get_model(modelA).graph.nodes if "add" in str(node)
        ]
    else:
        layersA = [layerA]
    if layerB is None:
        layersB = [
            str(node) for node in _get_model(modelB).graph.nodes if "add" in str(node)
        ]
    else:
        layersB = [layerB]

    comparator_instance = comparator.get_comparator()

    # We only need to instantiate the dataloader once; iterating over it once per job will re-use
    # the same data for all jobs. This is not *generally* true of shuffled dataloaders, but
    # _get_test_data uses a Subset of specifically the same indices over and over.
    data = _get_test_data(dataset, data_seed, m)

    jobs = [
        {
            "modelA": modelA,
            "layerA": lA,
            "modelB": modelB,
            "layerB": lB,
            "dataset": dataset,
            "comparator": comparator,
            "m": m,
            "data_seed": data_seed,
        }
        for lA, lB in itertools.product(layersA, layersB)
    ]
    index = RunIndex.from_experiment(jobs[0])
    for job in index.pending(jobs):
        print("Running", job)
        with logged_run(job):
            getterA = (
                _get_model(job["modelA"]).set_output(job["layerA"]).eval().to(device)
            )
            getterB = (
                _get_model(job["modelB"]).set_output(job["layerB"]).eval().to(device)
            )
            score = comparator_instance.streaming_compare(
                lambda: _get_reps_iter(data, getterA, getterB, device)
            )
            mlflow.log_metric("score", score.item())


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")
    parser.add_argument(
        "--mlflow_tracking_uri", default="sqlite:////projects/demo-similarity/mlflow.db"
    )
    parser.add_argument("--mlflow_experiment", default="debug")
    args = parser.parse_args()
    args.pop("config")

    # Set up MLflow. The '/projects/your-project-name/mlflow.db' path is the recommended place to
    # put your work on the shared lab server.
    mlflow.set_tracking_uri(args.pop("mlflow_tracking_uri"))
    mlflow.set_experiment(args.pop("mlflow_experiment"))

    main(**vars(parser.instantiate(args)))
