import mlflow
from nn_lib.utils import logged_run

from my_project import dothething


def main(a: int, b: int):
    result = dothething(a, b)
    mlflow.log_metric("result", result)


if __name__ == "__main__":
    import os

    import jsonargparse
    from dotenv import load_dotenv

    parser = jsonargparse.ArgumentParser(description="Run the dothething function.")
    parser.add_function_arguments(main)
    args = parser.parse_args()

    load_dotenv()  # loads .env file into environment variables

    user = os.getenv("MLFLOW_SQL_USERNAME")
    password = os.getenv("MLFLOW_SQL_PASSWORD")
    db_name = os.getenv("MLFLOW_SQL_DATABASE")

    mlflow.set_tracking_uri(f"mysql+pymysql://{user}:{password}@localhost:3306/{db_name}")
    if not mlflow.get_experiment_by_name("expt1"):
        mlflow.create_experiment("expt1", artifact_location="/projects/myproject/expt1/")
    mlflow.set_experiment("expt1")

    with logged_run(args):
        main(**args.as_dict())
