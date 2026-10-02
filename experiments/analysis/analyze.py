import argparse
import json
from pathlib import Path

from experiments.analysis.statistics import analyze_results


def main():
    parser = argparse.ArgumentParser(description="Analyze a completed experiment run.")
    parser.add_argument("results", type=Path, help="Path to the run's results.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = analyze_results(
        results,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = args.results.parent / "analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()