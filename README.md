# Procedural Graph Mesh for Team Agents

This project is a small prototype for a procedural graph system that watches agent work, turns successful experiences into reusable procedures, and lets future tasks pull back only the patterns that are relevant to their project and goal.

The code is intentionally compact. It is meant to show the shape of the system rather than act as a full production product.

## What it does

The service has four main parts:

- Agent adapters that collect a task trajectory.
- A learning pipeline that turns the raw trajectory into a candidate procedure.
- A validation step that prevents a single user from creating team-wide certainty too early.
- A project-scoped retrieval API that returns only relevant procedures.

## Project layout

- src/cls/agents: adapter interfaces for real and mock agents
- src/cls/ingestion: trajectory schemas and storage helpers
- src/cls/learning: analysis, extraction, validation, and update steps
- src/cls/graph: in-memory graph model and retrieval API
- src/cls/service: async worker and FastAPI endpoints
- experiments: benchmark runner and config examples
- tests: verification for validation and leakage rules

## Quick start

From the project root:

```bash
cd central-learning-service
PYTHONPATH=src ./venv/bin/pytest -q
```

If you want to run the example benchmark script:

```bash
cd central-learning-service
PYTHONPATH=src python experiments/run.py --config experiments/configs/pilot.yaml
```

## Design notes

The key rule is that a single successful run does not become shared team knowledge automatically. The validator keeps confidence low until there is repeated evidence or a second user confirms the same pattern.

The graph is also namespace-scoped. Procedures are only returned when the current project matches the scope of the stored procedure, which keeps unrelated projects isolated.

## Why this exists

The project is trying to test a simple idea: if agent activity is converted into reusable procedures, later work can benefit from that experience without simply replaying raw logs.

It is not a complete memory system or a production-grade graph database. It is a workable prototype for the core learning loop.
