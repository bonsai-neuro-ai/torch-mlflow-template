from my_project import dothething


def main(a: int, b: int):
    result = dothething(a, b)
    print(f"The result of dothething({a}, {b}) is: {result}")


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser(description="Run the dothething function.")
    parser.add_function_arguments(main)
    args = parser.parse_args()
    main(**args.as_dict())
