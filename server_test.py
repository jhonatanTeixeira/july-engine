"""
Prova de conceito: dispatcher single-thread com round-robin token-a-token.

Nenhum lock em volta da resposta inteira. Uma única coroutine (o
dispatcher) é a ÚNICA coisa que toca a instância `Llama`. Ela roda em loop
e, a cada volta, dá EXATAMENTE um passo (eval + sample de 1 token) pra
cada sessão ativa, antes de passar pra próxima. Isso intercala N gerações
de verdade (cada cliente recebe tokens em paralelo, não em blocos), sem
nunca ter duas chamadas nativas ao llama.cpp simultâneas — porque só
existe UMA thread chamando eval()/sample(), nunca duas.

Rodar:
    .venv/bin/uvicorn server_test:app --host 0.0.0.0 --port 3001
"""
import asyncio
import os
import time
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from llama_cpp.llama import active_seq_id

MODEL_PATH = os.environ.get(
    "SERVER_TEST_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/"
        "snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
    ),
)
N_SEQ_MAX = 2

app = FastAPI()
llm: Llama | None = None


@dataclass
class Session:
    seq_id: int
    prompt_tokens: list[int]
    max_tokens: int
    prefilled: bool = False
    generated: int = 0
    done: bool = False
    error: str | None = None
    tokens_out: list[int] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)


sessions: dict[int, Session] = {}
_free_seq_ids: asyncio.Queue[int] = asyncio.Queue()
_dispatcher_wakeup = asyncio.Event()
_dispatcher_task: asyncio.Task | None = None


@app.on_event("startup")
async def load_model():
    global llm, _dispatcher_task
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_seq_max=N_SEQ_MAX,
        n_batch=2048,
        offload_kqv=True,
        kv_unified=True,
        flash_attn=True,
        verbose=False,
    )
    for i in range(N_SEQ_MAX):
        _free_seq_ids.put_nowait(i)
    _dispatcher_task = asyncio.create_task(dispatcher())
    print(f"server_test: modelo carregado (n_seq_max={N_SEQ_MAX}), dispatcher iniciado", flush=True)


def _step(session: Session) -> int | None:
    """Executa UM passo de geração para uma sessão. Só o dispatcher chama isto."""
    active_seq_id.set(session.seq_id)

    if not session.prefilled:
        llm.reset(seq_id=session.seq_id)
        llm.eval(session.prompt_tokens, seq_id=session.seq_id)
        session.prefilled = True
    else:
        llm.eval([session.tokens_out[-1]], seq_id=session.seq_id)

    return llm.sample(temp=0.0)


async def dispatcher():
    """Loop single-thread: round-robin de 1 passo por sessão ativa por volta."""
    while True:
        active = [s for s in sessions.values() if not s.done]
        if not active:
            _dispatcher_wakeup.clear()
            await _dispatcher_wakeup.wait()
            continue

        for session in active:
            if session.done:
                continue
            try:
                token = _step(session)
            except Exception as e:
                session.error = str(e)
                session.done = True
                session.event.set()
                continue

            session.tokens_out.append(token)
            session.generated += 1

            if token == llm.token_eos() or session.generated >= session.max_tokens:
                session.done = True

            session.event.set()

        # Cede o controle pro event loop entre voltas (não pro llama.cpp — só uma
        # thread nunca chama decode() aqui, isto só deixa a ASGI responder outras coisas)
        await asyncio.sleep(0)


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 64


@app.get("/health")
async def health():
    return {"status": "ok", "n_seq_max": N_SEQ_MAX, "active_sessions": len(sessions)}


@app.post("/chat")
async def chat(req: ChatRequest):
    seq_id = await _free_seq_ids.get()
    try:
        prompt = llm.tokenize(
            f"<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n".encode(),
            add_bos=True,
            special=True,
        )
        session = Session(seq_id=seq_id, prompt_tokens=prompt, max_tokens=req.max_tokens)
        sessions[seq_id] = session
        _dispatcher_wakeup.set()

        t0 = time.monotonic()
        while not session.done:
            session.event.clear()
            await session.event.wait()

        if session.error:
            return {"seq_id": seq_id, "error": session.error}

        text = llm.detokenize(session.tokens_out).decode(errors="ignore")
        return {
            "seq_id": seq_id,
            "content": text,
            "tokens": session.generated,
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    finally:
        sessions.pop(seq_id, None)
        _free_seq_ids.put_nowait(seq_id)
