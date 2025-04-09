from nn_lib.models import get_pretrained_model, GraphModulePlus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print model metadata to stdout")
    parser.add_argument("model", help="Model name", type=str)
    parser.add_argument("--layers", action="store_true", help="Print layer names")
    args = parser.parse_args()

    model = get_pretrained_model(args.model)

    if args.layers:
        module_graph = GraphModulePlus.new_from_trace(model)
        for node in module_graph.graph.nodes:
            print(node.name)
