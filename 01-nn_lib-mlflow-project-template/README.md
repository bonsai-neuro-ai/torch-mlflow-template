# Project Template

This builds on `00-minimal-python-project-template` with a focus on using `mlflow` and our in-house
`nn_lib` tools.

## Setting up MLFlow tracking on the lab server

MLFlow tracks two kinds of information for us:

- A __database__ of experiments, runs, and metrics. This is where all the information about your 
  experiments is stored.
- __File storage__ also known as artifacts.

Both need to be configured.

### Option 1: `sqlite` backend

SQLite is good-enough for small amounts of tracking. It gives you full control over the database and
requires close to zero setup. It runs into issues, though, when it gets large and/or when multiple
jobs try to read or write at the same time.

Say we have a directory `/projects/myproject/` and want to track information in `/projects/myproject/mlflow.db` 
while storing artifacts per-experiment in directories like `/projects/myproject/expt1/`. A SQLite 
backend is configured in a script like this:

```python
import mlflow

def main():
    ... # log params, metrics, and artifacts here

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:////projects/myproject/mlflow.db")
    if not mlflow.get_experiment_by_name("expt1"):
        mlflow.create_experiment("expt1", artifact_location="/projects/myproject/expt1/")
    mlflow.set_experiment("expt1")

    with mlflow.start_run():
        main()
```

### Option 2: A full SQL database backend

Larger projects or projects with multiple users may want to use a full SQL backend. This requires
a bit more setup because SQL itself is managed on a Linux server by background services that require
admin privileges to configure.

As of October 2026, our lab server has a MariaDB server running on port 3306, and the only users with
admin privileges are Richard or GCCISIT. Ask Richard or GCCISIT to create a SQL user + password for
you (once ever), and to configure any new databases for you (once per project). Protect your username
and password by putting them in a `.env` file or config file that you never commit to git.

MLFlow needs some extra libraries to be able to talk to SQL databases, so you will need to add `pymysql`
to your project dependencies (see `pyproject.toml`).

Just like SQLite, you configure MLFlow tracking and artifact storage separately, like this:

```python
import mlflow

def main():
    ... # log params, metrics, and artifacts here

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()  # loads .env file into environment variables
    
    user = os.getenv("MLFLOW_SQL_USERNAME")
    password = os.getenv("MLFLOW_SQL_PASSWORD")
    db_name = os.getenv("MLFLOW_SQL_DATABASE")
    
    mlflow.set_tracking_uri(f"mysql+pymysql://{user}:{password}@localhost:3306/{db_name}")
    if not mlflow.get_experiment_by_name("expt1"):
        mlflow.create_experiment("expt1", artifact_location="/projects/myproject/expt1/")
    mlflow.set_experiment("expt1")

    with mlflow.start_run():
        main()
```
