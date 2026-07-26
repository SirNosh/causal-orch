# NATIVE_TYPED_TOOL_INTERFACE

This condition sends `return_artifact` to the pinned llama.cpp server as an
OpenAI-style function tool generated from `EvidenceReport.to_json_schema()`.
The existing Python validator remains authoritative and receives normalized
arguments without field renaming, unknown-field deletion, or semantic repair.

## Gate 1: native artifact only

Pinned 2,000-token, thinking-on result:

- Clean first-attempt validator acceptance: 20/20
- Repairs: 0/20
- Unknown fields: 0/20
- Missing fields: 0/20
- Malformed arguments: 0/20
- Budget exhaustion: 0/20

This passes Gate 1 and resolves the stock textual-interface schema ambiguity.

## Gate 2: real email tool plus native artifact

All conditions used fresh pinned Gaia2 environments, the real
`Emails__list_emails` tool, correct tool-call ID replay, and the original
2,000-token cumulative budget.

- Initial native loop: 1/5 passed. Four workers answered in plain text after
  the real tool instead of calling `return_artifact`.
- Fixed generic post-tool continuation: 4/5 passed.
- One deterministic format-only retry with remaining-token enforcement: 3/5
  passed. One failure consumed the full remaining budget before completing the
  forced native call; another still returned text with 484 tokens remaining.

The final Gate 2 condition therefore does not meet the required 5/5 threshold.
The failure is no longer artifact-schema presentation. It is unreliable
post-tool native-call completion under the frozen cumulative budget.

Gate 3 and the five-attempt full qualification were not run because Gate 2 did
not pass.

Full outgoing requests and raw responses are preserved in the local ignored
gate directories. They are not included in Git because they contain full model
reasoning and benchmark observations; the committed summaries contain the
normalized calls, validator results, and aggregate outcome.
