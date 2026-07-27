# causal-orch

A minimal kernel for one causal question: after the orchestrator's first valid
delegation proposal, what changes when that delegation is randomly executed or
suppressed?

The package uses Gaia2 only for scenario setup, tools, application state, and
native validation. Model actions are provider-native function calls:

- an ordinary Gaia2 tool;
- `delegate(objective)`;
- `final_answer(answer)`.

The first eligible `delegate` call reveals one concealed assignment. Treatment
runs a fresh worker with only Gaia2 tools explicitly marked
`write_operation=False`. Control returns a fixed suppression observation. The
same orchestrator loop then continues.

## Run

Install the locked Python 3.11 environment:

```powershell
uv sync --locked --extra test
```

Set a fixed OpenAI-compatible endpoint and model:

```powershell
$env:CAUSAL_ORCH_BASE_URL = "http://127.0.0.1:1234/v1"
$env:CAUSAL_ORCH_MODEL = "your-loaded-model-id"
```

Then pass one or more local Gaia2 scenario JSON files:

```powershell
uv run python scripts/smoke.py --scenario C:\path\scenario.json
uv run python scripts/proposal_pilot.py --scenario C:\path\scenario.json
uv run python scripts/randomized_pilot.py --scenario C:\path\scenario.json
```

For a remote endpoint, set `OPENAI_API_KEY` or select another variable with
`--api-key-env`. JSONL traces are written under `artifacts/minimal` by default.
Raw provider responses, token counts, latency, tool calls, assignment, worker
completion, and Gaia2's binary result are retained.

`smoke.py` fixes assignment to execute, `proposal_pilot.py` fixes assignment to
suppress while measuring proposal behavior, and `randomized_pilot.py` uses a
reproducible 50/50 assignment derived from `--seed` and run ID.

## Verify

```powershell
uv run --locked pytest -q
```

The tests cover typed action decoding, exact tool dispatch, assignment
concealment, one intervention per run, suppression without worker launch,
structural worker isolation, worker state hashing, loop continuation, and native
Gaia2 validation.
