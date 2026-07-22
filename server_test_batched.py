"""
Protótipo do `DecodeGate`: fila + eleição de líder — substitui o modelo anterior de "rodadas
sincronizadas geridas por um dispatcher central" (versão antiga deste mesmo arquivo).

Corrige os bugs encontrados na revisão da versão anterior:
- `accept(token, False)` fixo -> `accept(token, grammar is not None)` (senão o parser da
  gramática nunca avança e a saída constrained-JSON/tool-call quebra depois do 1º token).
- Reamostragem espúria do último token do prompt no prefill (que deslocava o KV cache por +1)
  -> eliminada por construção: prefill e geração em regime estacionário usam o MESMO mecanismo
  ("submeta N tokens, peça logits só do último"), então o primeiro token é sempre amostrado
  direto dos logits do próprio decode que processou o prompt, sem reinjeção.

Mecanismo (fila + líder, não rodada sincronizada):
- Quem chama `gate.submit(req)` e encontra a fila livre (ninguém decodificando agora) vira
  líder: drena TUDO que estiver esperando naquele instante (não só quem chegou primeiro),
  monta UM `LlamaBatch` com as contribuições de todos, chama `decode()` uma vez, distribui os
  resultados (via `asyncio.Event`), e só então recheca a fila antes de largar a liderança.
- Quem chega enquanto já tem líder ativo só entra na fila e espera o próprio evento — nunca
  processa por conta própria (evita rodar o mesmo lote em duplicidade).
- Sem `sleep()` artificial: o líder processa imediatamente quem estiver esperando, seja 1 ou N.
  O `asyncio.sleep(0)` entre rodadas serve só pra dar a chance de quem acabou de ser liberado
  rodar seu próprio próximo passo (amostrar + submeter o próximo token) antes de rechecar a fila.

Rodar:
    .venv/bin/uvicorn server_test_batched:app --host 0.0.0.0 --port 3002
"""
import asyncio
import os
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from llama_cpp.llama import active_seq_id
from llama_cpp._internals import LlamaSamplingContext, LlamaSamplingParams

MODEL_PATH = os.environ.get(
    "SERVER_TEST_MODEL",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/"
        "snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
    ),
)
N_SEQ_MAX = 3

app = FastAPI()
llm: "Llama | None" = None
gate: "DecodeGate | None" = None


@dataclass
class DecodeRequest:
    """Um pedido de decode: adicionar `tokens` ao `seq_id`, a partir da posição `pos`, e
    devolver o índice de logits do ÚLTIMO token (pra amostrar o próximo token dessa sessão).
    Prefill (muitos tokens) e geração em regime estacionário (1 token) usam a MESMA classe —
    só varia o tamanho de `tokens`."""
    seq_id: int
    tokens: list[int]
    pos: int
    result_idx: int = -1
    error: str | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


class DecodeGate:
    """Fila + líder. Só quem acha a fila vazia processa; quem chega depois só espera."""

    def __init__(self, llm: Llama):
        self.llm = llm
        self._pending: list[DecodeRequest] = []
        self._leader_active = False

    async def submit(self, req: DecodeRequest) -> int:
        self._pending.append(req)
        if not self._leader_active:
            self._leader_active = True
            try:
                await self._lead()
            finally:
                self._leader_active = False
        else:
            await req.event.wait()

        if req.error:
            raise RuntimeError(req.error)
        return req.result_idx

    async def _lead(self):
        while self._pending:
            batch_reqs = self._pending
            self._pending = []

            try:
                batch = self.llm._batch
                batch.reset()
                batch_pos = 0  # posição BRUTA dentro do batch (não um contador de pedidos —
                                # get_logits_ith() espera a posição real, não um índice compactado)
                for r in batch_reqs:
                    active_seq_id.set(r.seq_id)
                    pos = r.pos
                    last_i = len(r.tokens) - 1
                    for i, tok in enumerate(r.tokens):
                        is_last = (i == last_i)
                        batch.add_token(tok, pos, [r.seq_id], is_last)
                        pos += 1
                        if is_last:
                            r.result_idx = batch_pos
                        batch_pos += 1

                print(
                    f"DecodeGate: 1 decode() cobrindo {len(batch_reqs)} pedido(s) "
                    f"(seq_ids={[r.seq_id for r in batch_reqs]}, batch.n_tokens={batch.n_tokens()})",
                    flush=True,
                )
                ret = self.llm._ctx.decode(batch)
                if ret != 0:
                    raise RuntimeError(f"llama_decode retornou {ret} (sem slot de KV cache disponível)")
            except Exception as e:
                for r in batch_reqs:
                    r.error = str(e)
                    r.event.set()
                continue

            for r in batch_reqs:
                r.event.set()

            # Dá a chance de quem acabou de ser liberado rodar seu próprio próximo passo
            # (amostrar + submeter o próximo token) antes de rechecar a fila. Sem isso, o
            # líder poderia nunca ceder o controle e travar as outras sessões pra sempre.
            await asyncio.sleep(0)


@dataclass
class Session:
    seq_id: int
    pos: int = 0
    sampling_ctx: "LlamaSamplingContext | None" = None
    tokens_out: list[int] = field(default_factory=list)


_free_seq_ids: "asyncio.Queue[int]" = asyncio.Queue()


@app.on_event("startup")
async def load_model():
    global llm, gate
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
    gate = DecodeGate(llm)
    for i in range(N_SEQ_MAX):
        _free_seq_ids.put_nowait(i)
    print(
        f"server_test_batched: modelo carregado (n_seq_max={N_SEQ_MAX}), "
        "DecodeGate (fila+líder) pronto",
        flush=True,
    )


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 64
    temperature: float = 0.0
    grammar: "str | None" = None  # GBNF, só pra validar o fix do accept()


@app.get("/health")
async def health():
    return {"status": "ok", "n_seq_max": N_SEQ_MAX}


@app.post("/chat")
async def chat(req: ChatRequest):
    seq_id = await _free_seq_ids.get()
    try:
        active_seq_id.set(seq_id)
        llm.reset(seq_id=seq_id)

        prompt_tokens = llm.tokenize(req.message.encode(), add_bos=True, special=True)
        session = Session(seq_id=seq_id)

        grammar_str = req.grammar or ""
        has_grammar = bool(grammar_str)
        params = LlamaSamplingParams(temp=req.temperature, grammar=grammar_str)
        session.sampling_ctx = LlamaSamplingContext(params, llm._model)

        # Prefill: submete o prompt inteiro de uma vez, pede logits só do último token —
        # o MESMO mecanismo da geração em regime estacionário, sem caminho especial.
        dreq = DecodeRequest(seq_id=seq_id, tokens=prompt_tokens, pos=session.pos)
        session.pos += len(prompt_tokens)
        idx = await gate.submit(dreq)

        token = session.sampling_ctx.sample(llm._ctx, idx=idx)
        session.sampling_ctx.accept(token, has_grammar)
        session.tokens_out.append(token)

        while len(session.tokens_out) < req.max_tokens and token != llm.token_eos():
            dreq = DecodeRequest(seq_id=seq_id, tokens=[token], pos=session.pos)
            session.pos += 1
            idx = await gate.submit(dreq)

            token = session.sampling_ctx.sample(llm._ctx, idx=idx)
            session.sampling_ctx.accept(token, has_grammar)
            session.tokens_out.append(token)

        text = llm.detokenize(session.tokens_out).decode(errors="ignore")
        return {"seq_id": seq_id, "content": text, "tokens": len(session.tokens_out)}
    finally:
        _free_seq_ids.put_nowait(seq_id)
