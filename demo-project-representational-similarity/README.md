# Demo representational similarity comparison

This is a partially outdated example.

__What to look for in this example:__

- The key idea of separating computation code (`compare_layers.py`) from plotting code (`plot_cka.py`).
- Usage of a shell script to invoke the py scripts
- Use of `mlflow` for parameter and metric tracking
- Use of `pandas` and `seaborn` to turn logged results into pretty plots 
- Use of some `nn_lib` tools for working with `mlflow`

__Known issues:__

- Artifact storage is not configured properly in this example.
- This example uses `requirements.txt` rather than `pyproject.toml` and doesn't use the other nice
  project structure best-practices that are outlined in the `00-` and `01-` examples. The code here
  is still valid and can be a useful reference for some things.

__TO-DO:__ update this example using structure from `00-` and `01-` examples.