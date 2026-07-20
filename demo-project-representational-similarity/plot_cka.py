import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from nn_lib.utils import search_runs_by_params

mlflow.set_tracking_uri("sqlite:////projects/demo-similarity/mlflow.db")

results = search_runs_by_params(
    experiment_name="demo-similarity",
    finished_only=True,
    params={"m": 1000, "comparator.class_path": "nn_lib.analysis.similarity.LinearCKA"},
)
assert len(results) > 0

# %%

def _layer_sort_key(layer):
    try:
        str_part, num_part = layer.rsplit("_", 1)
        return str_part, int(num_part)
    except ValueError:
        return layer, 0

for group, df in results.groupby(["params.modelA", "params.modelB"]):
    # Convert df into a table where rows are layerA and columns are layerB
    # and values are the CKA score
    df = df.pivot(
        index="params.layerA",
        columns="params.layerB",
        values="metrics.score",
    )

    # Sort the index and columns
    df = df.reindex(
        sorted(df.index, key=_layer_sort_key),
        axis=0,
    ).reindex(
        sorted(df.columns, key=_layer_sort_key),
        axis=1,
    )

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=False,
        cmap="magma",
        cbar_kws={"label": "CKA Score"},
    )
    plt.ylabel(group[0])
    plt.xlabel(group[1])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
