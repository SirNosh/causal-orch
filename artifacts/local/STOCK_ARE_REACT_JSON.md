# STOCK_ARE_REACT_JSON

This archived diagnostic condition uses stock ARE's textual JSON-ReAct
interface. `return_artifact` is presented as an unconstrained `artifact: any`
value, while the strict `EvidenceReport` contract appears only in prompt prose.

Pinned Nanbeige4.2-3B Q8_0 results:

- Frozen 2,000-token condition: 0/20 clean first attempts, 12/20 eventual
  validator acceptances, and 8/20 budget exhaustions.
- Diagnostic 4,000-token condition: 0/20 clean first attempts, 18/20 eventual
  validator acceptances, and 2/20 budget exhaustions.
- Diagnostic 2,000-token thinking-off condition: 0/20 clean first attempts,
  8/20 eventual validator acceptances, and 12/20 budget exhaustions.

All 60 first attempts used an unknown `type` field and omitted the required
`artifact_type` field. These results remain a diagnostic comparison and do not
define the native typed-tool condition.
