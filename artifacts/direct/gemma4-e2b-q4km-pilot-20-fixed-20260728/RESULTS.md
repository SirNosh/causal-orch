# Gemma 4 E2B direct baseline

## Frozen condition

- Model: `google/gemma-4-e2b`
- Quantization: `Q4_K_M`
- Runtime: LM Studio at `http://127.0.0.1:1234/v1`
- Loaded context: 131,072 tokens
- Delegation: unavailable (ordinary Gaia2 tools plus `final_answer`)
- Orchestrator steps: 20 maximum
- Orchestrator output: 8,192 tokens maximum per call
- Transport timeout: 300 seconds
- Simulated model time: 5 seconds per call
- Temperature: 0
- Scenarios: six local Gaia2 scenarios cycled across 20 fresh runs

The separate fixed preflight completed five valid model turns, four ordinary
tool calls, native validation, and a final answer without a timeout or
structured-action failure.

## Results

| Metric | Result |
| --- | ---: |
| Gaia2 success | 0 / 20 |
| Native validation reached | 20 / 20 |
| Final answers produced | 10 / 20 |
| Task-completion failures | 10 / 20 |
| Structured-action failures | 7 |
| Invalid Gaia2 tool actions | 3 |
| Model calls | 55 |
| Completed model responses | 48 |
| Ordinary tool calls | 35 |
| Input tokens | 714,921 |
| Output tokens | 48,238 |
| Model latency | 369.146 seconds |
| Aggregate run wall time | 782.625 seconds |
| Simulated model time | 275 seconds |
| Effective completion throughput | 130.67 tokens/second |
| Maximum successful call latency | 27.139 seconds |
| Transport timeouts | 0 |

The seven structured failures returned no function call despite required
tool choice. Three runs attempted to remove an apartment that was not saved.
The ten final answers all failed Gaia2 validation. LM Studio returned an empty
`stats` object, so prompt-evaluation time was not available separately; the
reported throughput uses end-to-end model-call latency and therefore includes
prompt processing.

The assignment manifest and all 20 JSONL traces are stored in this directory.
The fixed preflight manifest and trace are in the adjacent
`gemma4-e2b-q4km-preflight-fixed-20260728` directory.
