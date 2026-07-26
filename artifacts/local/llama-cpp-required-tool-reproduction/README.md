# llama.cpp `tool_choice: required` reproduction

Runtime:

```text
Repository: https://github.com/Nanbeige/llama.cpp
Branch: nanbeige42
Commit: d28da865bf284acaecc98ad18a3c1f607c0fd754
Build: b10042-d28da865b
GGUF: Nanbeige4.2-3B-Q8_0.gguf
GGUF SHA-256: 76627e550979d8ea5746cb11922ad10352d91546aa540c0e8292522f8dd9c2b5
Server SHA-256: ffd50a1818124aec857b9bedee3dadcf905548f7fc83a9c06dcdaa281d8effab
```

Launch:

```powershell
$env:LLAMA_ARG_LOG_FILE=".local\llama-required-repro-verbose.log"
$env:LLAMA_ARG_LOG_VERBOSITY="100"
.\scripts\run_local_server.ps1
```

Requests:

```bash
curl -sS -o auto-response.json -w "HTTP %{http_code}\n" \
  -H "Content-Type: application/json" \
  --data-binary @auto-request.json \
  http://127.0.0.1:8080/v1/chat/completions

curl -sS -o required-response.json -w "HTTP %{http_code}\n" \
  -H "Content-Type: application/json" \
  --data-binary @required-request.json \
  http://127.0.0.1:8080/v1/chat/completions
```

Observed:

```text
auto:     HTTP 200, generated a parsed ping tool call
required: HTTP 400, generation did not start
```

The complete 400 response is in `required-response.json`. The relevant verbose
sampler initialization log is in `grammar-init-error.log`.
