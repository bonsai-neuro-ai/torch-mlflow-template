import mlflow
import torch
from nn_lib.datasets import ImageNetDataModule
from nn_lib.models import get_pretrained_model, GraphModulePlus
from nn_lib.analysis.similarity import LinearCKA
from nn_lib.utils import search_runs_by_params, log_params_and_config
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


def run_compare(
    modelA: str,
    layerA: str,
    modelB: str,
    layerB: str,
    dataset: str,
    comparator: LinearCKA,
    m: int = 1000,
    data_seed: int = 189645,
    device: str | torch.device = "cuda",
):
    getterA = (
        GraphModulePlus.new_from_trace(get_pretrained_model(modelA))
        .squash_all_conv_batchnorm_pairs()
        .set_output(layerA)
        .eval()
        .to(device)
    )
    getterB = (
        GraphModulePlus.new_from_trace(get_pretrained_model(modelB))
        .squash_all_conv_batchnorm_pairs()
        .set_output(layerB)
        .eval()
        .to(device)
    )

    data = _get_test_data(dataset, data_seed, m)

    @torch.no_grad()
    def _iter_batches():
        for batch, _ in data:
            batch = batch.to(device, non_blocking=True)
            yield getterA(batch), getterB(batch)

    score = comparator.streaming_compare(_iter_batches)
    mlflow.log_metric("score", score.item())


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(run_compare)
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

    # Check if a run with the same parameters already exists. If so, exit early to avoid
    # duplication. NOTE: if run on a cluster, this is maybe a bad pattern because we need to wait
    # for the job to be scheduled and run before we are even able to check if it needs running. A
    # to-do item is to write a demo where the 'exit-early' pattern happens before the job is
    # submitted.
    existing_runs = search_runs_by_params(
        params=args, finished_only=True, skip_keys=["device"]
    )
    if not existing_runs.empty:
        print("Run already exists with these params. Skipping.")
        exit(0)

    # Start the run and log params
    args_instantiated = parser.instantiate(args)
    with mlflow.start_run() as run:
        mlflow.log_text(parser.dump(args), "config.yaml")
        log_params_and_config(args, parser)
        run_compare(**args_instantiated.as_dict())
