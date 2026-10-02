# Procedural Graph Mesh for Team Agents

This repository contains an experimental service and harness for studying whether coding-agent experience can be turned into project-scoped procedures that help later agents. It is research infrastructure in progress, not a production-ready service or evidence that the hypothesis is true.

## What is implemented

- Versioned trajectories with stable IDs and linked procedure evidence.
- SQLite-backed trajectory archival and retryable job state.
- A background learning worker and SQL-persisted procedure graph.
- Retrieval filtered by project, exact repository revision, validation state, and independent source evidence.
- Retrieval exposure logs that store procedure IDs and a task hash rather than task prompt text.
- A four-condition experiment runner: independent, shared history, static repository graph, and procedural graph.
- Per-task worktrees, objective command checks, run manifests, raw artifacts, metrics, and paired bootstrap summaries.

The extractor currently preserves observed action sequences and the validator uses exact-match rules. These are baselines for experimentation, not mature generalization or knowledge-quality solutions.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run the tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Run the API locally:

```bash
PYTHONPATH=src .venv/bin/uvicorn cls.service.api:app --reload
```

The service uses `DATABASE_URL` when set; otherwise it creates `procedural_graph_mesh.db` in the current directory.

## Experiment harness

The bundled configuration is only a deterministic smoke test. It checks that the runner, worktrees, four conditions, and artifact writing function; it does not measure agent performance and must not be cited as research evidence.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.run --config experiments/configs/pilot.yaml
```

A run is written beneath `experiments/runs/`. It includes the config and manifest, task manifest, per-task trajectories and outcomes, candidate procedures, learned graphs, retrieval events, logs, `results.json`, `metrics.csv`, and `analysis.json`.

For a research run, configure a pinned agent container image and a digest-pinned oracle image. Both run without network access; the agent receives only its task worktree, while the oracle receives a read-only view. Agent commands receive a JSON object on stdin containing the task, memory context, and seed, and must return a JSON object on stdout. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the research protocol and publication gates.

The reported transfer measure is based on procedure IDs an agent says it used. Exposure or self-reported use is not, by itself, causal evidence of benefit. The primary comparison should be objective task success between controlled arms, with uncertainty and limitations reported.

## Current limits

The repository does not yet include a real model-backed agent integration, a human-labeled extraction-quality set, a powered confirmatory task set, schema migrations, authentication, or operational security controls. The current implementation should not be described as production-ready or as demonstrating a positive research result.
