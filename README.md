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
`write_operation=False`, excluding user communication, system, reminder,
notification, time-control, and generic execution tools. Control returns a
fixed, strategy-neutral status object. The same orchestrator loop then
continues.

Assignments are written before a batch starts to a small JSON manifest keyed by
`batch_id:scenario_id:repetition`. Retrying a unit creates a new run and
incremented attempt while preserving its original assignment. Randomized
manifests are deterministically balanced within the batch.

Gaia2 time is paused during every model call. Orchestrator calls resume with a
fixed five-second simulated offset; a worker stays paused for its entire episode
and resumes once with five seconds per worker call. Provider wall latency
therefore cannot become part of treatment.

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
uv run python scripts/direct_baseline.py --scenario C:\path\scenario.json
uv run python scripts/proposal_pilot.py --scenario C:\path\scenario.json
uv run python scripts/randomized_pilot.py --scenario C:\path\scenario.json
```

`direct_baseline.py` removes the delegation function and delegation language
entirely, leaving ordinary Gaia2 tools plus `final_answer`.
Orchestrator calls have a fixed 8,192-token output ceiling. Use
`--timeout-seconds` to account for local prompt processing and generation
without changing simulated model time.

Use a stable batch name when runs may be retried. Attempt IDs are inferred from
existing traces, or can be supplied explicitly:

```powershell
uv run python scripts/randomized_pilot.py `
  --batch-id proposal-pilot-01 `
  --scenario C:\path\scenario.json
```

The forced-delegation integration check can be run under both assignments:

```powershell
uv run python scripts/smoke.py --scenario C:\path\scenario.json `
  --force-delegation-objective "Find the relevant fact" --assignment execute
uv run python scripts/smoke.py --scenario C:\path\scenario.json `
  --force-delegation-objective "Find the relevant fact" --assignment suppress
```

For a remote endpoint, set `OPENAI_API_KEY` or select another variable with
`--api-key-env`. JSONL traces are written under `artifacts/minimal` by default.
Raw provider responses, token counts, latency, tool calls, assignment, worker
completion, state hashes, canonical and executed actions, stable identifiers,
typed user/environment/stop notifications, and Gaia2's binary result are
retained. Malformed provider responses retain their raw payload. Worker requests
include the remaining hard output-token limit.

`smoke.py` fixes assignment to execute, `proposal_pilot.py` fixes assignment to
suppress while measuring proposal behavior, and `randomized_pilot.py` uses a
reproducible 50/50 assignment derived from `--seed` and run ID.

## Verify

```powershell
uv run --locked pytest -q
```

The tests cover typed action decoding, exact tool dispatch, assignment
concealment, one intervention per run, suppression without worker launch,
structural worker isolation, worker state hashing, treatment-failure retention,
fresh environments, raw/canonical/executed trace alignment, forced treatment
and control continuation, stable retry assignments, fixed simulated model time,
typed notifications, hard worker token limits, and native Gaia2 validation.
Malformed actions fail the run under a fixed no-repair policy and never consume
an assignment.
