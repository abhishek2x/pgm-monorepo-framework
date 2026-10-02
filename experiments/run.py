import argparse

from experiments.runner import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a procedural-memory experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    args = parser.parse_args()

    run_dir = run_experiment(load_config(args.config))
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
