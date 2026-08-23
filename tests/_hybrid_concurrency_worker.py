"""
Standalone worker process for test_integration.py's
test_concurrent_hybrid_model_no_corruption_or_crash.

Why a separate process instead of an in-process pytest coroutine: the bug
this regression-tests (HybridCheckpointCache having zero thread-safety while
being invoked concurrently by every seq_id that hits a KV-cache prefix
mismatch — see llama_cache.py) manifests as a real C-level heap corruption
("double free or corruption (fasttop)") or a segfault. Either one aborts the
whole interpreter. If that happened inside the pytest process itself, it
would silently kill the entire test run (and every other test's result along
with it) instead of failing just this one test. Running the actual repro in
a child process lets the parent test observe "the child died from a signal"
as a normal, reportable test failure.

Boots the real FastAPI app in-process (same ASGITransport pattern the rest
of tests/test_integration.py uses) and drives it purely over real HTTP-shaped
requests — no mocking of the inference stack, per this repo's test
conventions.

Contract with the parent test: prints exactly one line starting with
"RESULT_JSON:" as the LAST line of stdout, containing a JSON object:
  - {"status": "skip", "reason": "..."}       -> parent calls pytest.skip()
  - {"status": "ok", "rounds": [...], "corrupted": [...]} -> parent asserts
    corrupted == []
If the process is killed by a signal before it manages to print that line,
there is no line to parse at all — the parent treats that (a nonzero /
signal returncode with no RESULT_JSON marker) as the crash itself.
"""
import asyncio
import json
import re
import sys

# Deliberately the same hybrid (qwen35 gated-delta-net) architecture used to
# validate MTP earlier, but with mtp_enabled=False: this bug predates MTP and
# is fully independent of it (the crash reproduces with load_mtp=False; the
# only thing that matters is the model being a hybrid/recurrent architecture
# that routes through _reuse_prefix_and_eval's HybridCheckpointCache branch).
MODEL_CATALOG_KEY = "unsloth/Qwen3.5-4B-MTP-GGUF"
MODEL_ROW = {
    "model_id": MODEL_CATALOG_KEY,
    "filename": "Qwen3.5-4B-Q4_K_M.gguf",
    # GGUF.load() sizes the actual native n_ctx as context_window * n_seq_max
    # (kv_unified) — keep this small enough that the VRAM estimate has
    # comfortable headroom under ordinary system fluctuation. Confirmed the
    # hard way: at context_window=1536 (n_ctx=4608 total), a run that landed
    # with slightly less free VRAM than usual tripped the orchestrator's
    # decrement_layers() fallback, forcing one layer onto CPU — which then
    # segfaulted on ITS OWN, unrelated to anything this test targets ("fused
    # Gated Delta Net (chunked) not supported" for a split CPU/GPU
    # placement on this hybrid architecture). That's a real, separate,
    # pre-existing issue worth its own investigation, but it isn't the bug
    # this test exists to catch — keep enough VRAM margin to reliably avoid
    # ever exercising that fallback path here.
    "context_window": 512,
    "n_seq_max": 3,
    "num_layers": -1,
    "mtp_enabled": False,
}

N_SEQ = 3
N_ROUNDS = 6
# This model always wraps its answer in a "Thinking Process:"/<think> preamble
# (see llama_gguf.py's reasoning_content splitting) before any post-</think>
# `content` — a small max_tokens cuts generation off mid-thought, leaving
# `content` legitimately empty even on a perfectly healthy round. Generous
# enough to reliably clear the preamble and reach real answer content.
MAX_TOKENS = 200

# Deliberately unrelated to each other so that every round, for every
# session, misses the KV-cache prefix match and forces the hybrid rollback
# (checkpoint save/find/restore) path in _reuse_prefix_and_eval — the exact
# path that calls into the previously-unlocked HybridCheckpointCache.
PROMPTS = [
    "The capital of France is Paris. The capital of Japan is",
    "Once upon a time there was a small dog named",
    "The three primary colors are red, blue, and",
    "In chemistry, the symbol for water is",
    "The opposite of hot is",
    "A group of wolves is called a",
    "The largest planet in the solar system is",
    "Roses are red, violets are",
]

# Flags a pathological short-substring repeat (e.g. corrupted KV state
# looping on a few tokens) — a real symptom seen while narrowing this bug,
# distinct from a merely-repetitive-but-coherent low-temperature completion.
_REPEAT_RE = re.compile(r"(.{1,6}?)\1{7,}")


def _looks_corrupted(text):
    if text is None:
        return "empty/None response"
    stripped = text.strip()
    if not stripped:
        return "empty response"
    if "�" in text:
        return "contains unicode replacement char (decode/encode boundary corruption)"
    m = _REPEAT_RE.search(text)
    if m:
        return f"pathological short-substring repeat of {m.group(1)!r}"
    return None


async def _one_call(client, session_id, prompt):
    return await client.post(
        "/v1/openai/chat/completions",
        json={
            "model": MODEL_CATALOG_KEY,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
        },
        headers={"x-session-id": session_id},
        timeout=120.0,
    )


def _emit(result):
    print(f"RESULT_JSON:{json.dumps(result)}")
    sys.stdout.flush()


async def main():
    from httpx import AsyncClient, ASGITransport
    from main import app
    from app.bridge import bridge
    from app.persistence import get_backend

    backend = get_backend()
    # Additive only: register the model in the catalog directly and address
    # it by that exact key in every request's "model" field, sidestepping
    # TEXT_PRESETS entirely so this can't interact with (or be broken by)
    # whatever shape other tests/fixtures leave that setting in.
    backend.set_model(MODEL_CATALOG_KEY, MODEL_ROW)

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            warm = await _one_call(client, "warmup-session", PROMPTS[0])
        except Exception as e:
            _emit({"status": "skip", "reason": f"model unavailable / failed to load: {e!r}"})
            return

        if warm.status_code != 200:
            _emit({
                "status": "skip",
                "reason": f"warmup call failed with status {warm.status_code}: {warm.text[:500]}",
            })
            return

        result = {"status": "ok", "rounds": [], "corrupted": []}

        for r in range(N_ROUNDS):
            # A FRESH session_id every round, not the same N_SEQ ids reused —
            # this forces SeqAllocator (only N_SEQ physical seq_id slots) to
            # recycle a physical slot from a DIFFERENT, now-abandoned
            # conversation onto each of these brand-new ones every round.
            # That's both a more faithful real-world pattern (many users,
            # few physical KV slots) than reusing the same N_SEQ ids forever,
            # and it keeps each conversation's own history at exactly one
            # round — no unbounded per-session growth to manage against the
            # deliberately small context_window above.
            session_ids = [f"hybrid-concurrency-r{r}-{i}" for i in range(N_SEQ)]
            responses = await asyncio.gather(
                *[
                    _one_call(client, session_ids[i], PROMPTS[(r * N_SEQ + i) % len(PROMPTS)])
                    for i in range(N_SEQ)
                ],
                return_exceptions=True,
            )
            round_texts = []
            for i, resp in enumerate(responses):
                if isinstance(resp, BaseException):
                    round_texts.append(f"<exception: {resp!r}>")
                    result["corrupted"].append({
                        "round": r, "session": session_ids[i],
                        "reason": f"request raised {resp!r}",
                    })
                    continue
                if resp.status_code != 200:
                    round_texts.append(f"<http {resp.status_code}: {resp.text[:300]}>")
                    result["corrupted"].append({
                        "round": r, "session": session_ids[i],
                        "reason": f"http {resp.status_code}: {resp.text[:300]}",
                    })
                    continue
                message = resp.json()["choices"][0]["message"]
                # `content` alone can be legitimately empty (still inside the
                # thinking preamble) while `reasoning_content` holds the real,
                # healthy output for this round — check the combination, only
                # treating BOTH as empty as an actual "empty response".
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                full_text = reasoning + content
                round_texts.append(content)
                reason = _looks_corrupted(full_text)
                if reason:
                    result["corrupted"].append({
                        "round": r, "session": session_ids[i],
                        "text": full_text[:300], "reason": reason,
                    })
            result["rounds"].append(round_texts)

        await bridge.stop()

    _emit(result)


if __name__ == "__main__":
    asyncio.run(main())
