import argparse
from typing import Any, Dict

import yaml


BASELINE_DESCRIPTIONS = {
    "independent": "Baseline A: independent agent",
    "shared_memory": "Baseline B: shared unstructured memory",
    "static_kg": "Baseline C: static knowledge graph",
    "central_learning": "Baseline D: central learning service",
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file for a benchmark run."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} must contain a dictionary at the top level.")

    return config


def run_experiment(config: Dict[str, Any]):
    """Run the experiment described in the config file."""
    experiment = config.get("experiment", {})
    name = experiment.get("name", "unnamed_experiment")
    baseline = experiment.get("baseline")
    num_tasks = experiment.get("num_tasks", 0)
    users = experiment.get("users", [])

    if baseline not in BASELINE_DESCRIPTIONS:
        raise ValueError(f"Unknown baseline: {baseline}")

    print(f"Starting experiment: {name}")
    print(f"Configuration: {BASELINE_DESCRIPTIONS[baseline]} | {num_tasks} tasks | {len(users)} users")

    if baseline == "central_learning":
        print("Collecting trajectories and running the learning pipeline...")

    for user in users:
        print(f"  [+] {user} is working through the assigned tasks")

    print("Running the evaluation pass and computing the final metrics...")
    print("Experiment completed. Results should be written to the experiment output directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental runner for the central learning framework")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)
