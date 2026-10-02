# Aria - AI Voice CX Agent for Aura Skincare (Python)

Browser voice agent: speak -> an LLM (with tool calling + brand guardrails) -> spoken reply, then a transcript and JSON call outcome.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # add your free Groq key from console.groq.com
uvicorn app:app --reload    # open http://localhost:8000 in Chrome/Edge
```
## Deploy (Render / Railway)
Build: `pip install -r requirements.txt` | Start: `uvicorn app:app --host 0.0.0.0 --port $PORT` | Env: `LLM_API_KEY`, `LLM_BASE_URL`, `MODEL`. HTTPS is required for mic access (these platforms provide it).

## Architecture
```
Browser mic -> Web Speech API (STT, en-IN) -> POST /chat (NDJSON stream)
FastAPI -> LLM (system prompt + get_order_details tool loop) -> streamed tokens
Browser splits stream into sentences -> SpeechSynthesis (en-IN voice) starts speaking on the FIRST sentence
End call -> POST /summary -> LLM JSON mode -> transcript + JSON
```
- **Tool calling:** the LLM decides when to call `get_order_details` or `cancel_order` (cancel only after the customer confirms; eligibility is re-checked in code and cancellations are scoped to the call session, so the mock DB never changes); the tool normalises IDs ("ord 101" -> ORD-101), returns `found:false` for unknown IDs, and computes `cancellable` / `return_window_open` / `shipping_fee` in code, so policy decisions on order data are deterministic, not LLM guesses.
- **Guardrails:** policies live in the system prompt as the only allowed facts; rules forbid promises outside policy, refuse off-topic requests, and require honesty when info is missing. Opened-item questions are asked, since the DB can't know.
- **Graceful degradation:** garbled STT -> ask to repeat; unknown/missing ID -> ask to verify; backend error -> spoken apology, call continues.
- **Latency:** fast Groq inference (openai/gpt-oss-120b with low reasoning effort), streamed output, sentence-level TTS, so speech starts before the full answer exists.
- **Barge-in:** speaking over Aria cancels TTS and aborts the stream (toggle off when using speakers to avoid echo).

## Tell us how you think
1. **Stack:** A modular STT -> LLM -> TTS pipeline is transparent and easy to explain and debug. Python/FastAPI keeps everything in one language with streaming and tool use built in. Browser STT/TTS means zero extra vendors, no audio infra, and near-instant startup, at the cost of Chrome/Edge-only and less natural voices.
2. **Hardest part:** Feeling natural: latency and interruption. Solved with streaming plus sentence-chunked TTS and barge-in that cancels speech and the in-flight request. Second: trusting noisy transcripts, handled with read-back of IDs and "ask to repeat" rules.
3. **One more week:** Replace browser STT/TTS with server-side Deepgram/Whisper streaming plus a neural Indian voice (ElevenLabs, or `edge-tts` en-IN-NeerjaNeural) over WebSockets with VAD, for consistent quality, better Hinglish, and real echo cancellation. Then add an automated eval suite of scripted policy conversations.
4. **At 1,000 calls/day:** Persistent sessions (Redis/Postgres) instead of in-memory; a real orders API with authentication and customer verification; observability (latency per stage, tool errors, cost); rate limiting and retries/fallback models; human handoff for angry or unresolved calls; PII redaction and consent notice; prompt-regression evals; and autoscaled workers with prompt caching to control cost.

## Known limits
In-memory sessions (lost on restart); Web Speech API needs Chrome/Edge and internet; refunds and returns are explained, not executed (cancellation is mocked per call). Free-tier LLM rate limits apply.
