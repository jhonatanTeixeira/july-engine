import os

import pytest

from llama_gguf.resource_calculator import estimate_vram_ram

# Real model used to validate MTP + concurrent multi-seq batching. Requires no
# network: skips cleanly if it isn't already cached locally, per this repo's
# "no mocking of the inference stack" convention for anything model-related.
_MODEL_ID = "unsloth/Qwen3.5-4B-MTP-GGUF"
_FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"


def _local_model_path() -> str | None:
    os.environ.setdefault("HF_HOME", ".huggingface")
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=_MODEL_ID, filename=_FILENAME, local_files_only=True)
    except Exception:
        return None


@pytest.mark.gpu
@pytest.mark.anyio
async def test_mtp_vram_estimate_accounts_for_n_rs_seq_rollback_buffer():
    """
    Regression test for a real gap found while validating MTP support:
    `estimate_vram_ram()`'s `mtp_enabled` estimate only accounted for the
    small second MTP context's own KV cache/compute buffer — it completely
    missed `Llama.__init__`'s native `n_rs_seq` recurrent-state rollback
    buffer (see `llama-memory-recurrent.cpp`), which `load_mtp=True` always
    requests for a hybrid/recurrent architecture (qwen35's gated-delta-net
    layers included) and which scales with `n_seq_max`. Missing it meant
    `n_seq_max=4 + mtp_enabled=True` silently overflowed a real 4GB GPU's
    VRAM budget at context-create time (a raw, unhelpful native CUDA OOM)
    instead of being caught here, before ever attempting to load.

    Confirmed against this exact model+hardware pairing: `n_seq_max=4 +
    mtp_enabled=True` genuinely does not fit in 4GB VRAM (measured via a
    real load attempt), while `n_seq_max=2 + mtp_enabled=True` genuinely
    does. This test pins the estimator to keep agreeing with that reality.
    """
    model_path = _local_model_path()
    if not model_path or not os.path.exists(model_path):
        pytest.skip(f"{_MODEL_ID} not cached locally — skipping VRAM estimate regression test")

    common = dict(model_path=model_path, context_window=1024, gpu_layers=-1, kv_unified=True)

    no_mtp = await estimate_vram_ram(**common, n_seq_max=4, mtp_enabled=False)
    assert no_mtp["mtp_rs_rollback_vram_mb"] == 0, "no rollback buffer should be budgeted when MTP is off"

    seq2_mtp = await estimate_vram_ram(**common, n_seq_max=2, mtp_enabled=True)
    seq4_mtp = await estimate_vram_ram(**common, n_seq_max=4, mtp_enabled=True)

    assert seq2_mtp["is_hybrid_recurrent"] is True, "qwen35 must be detected as a hybrid/recurrent architecture"
    # The whole point of this fix: the rollback buffer must scale with
    # n_seq_max, and must not be zero for a hybrid architecture with MTP on.
    assert seq2_mtp["mtp_rs_rollback_vram_mb"] > 0
    assert seq4_mtp["mtp_rs_rollback_vram_mb"] > seq2_mtp["mtp_rs_rollback_vram_mb"]

    # The actual regression this exists to catch: at n_seq_max=4 the estimate
    # must now land at/over a 4GB card's real usable budget (confirmed via an
    # actual load attempt to OOM at context-create time), while n_seq_max=2
    # must stay comfortably under it (confirmed via an actual successful
    # load + generation). A small safety margin below the raw 4096 MiB card
    # size accounts for the driver/desktop-compositor baseline usage that's
    # never available to begin with.
    USABLE_4GB_CARD_MB = 3800
    assert seq4_mtp["total_vram_mb"] >= USABLE_4GB_CARD_MB, (
        "n_seq_max=4 + mtp_enabled=True should be estimated at/over a 4GB card's "
        f"usable budget (matches the real OOM) — got {seq4_mtp['total_vram_mb']} MiB"
    )
    assert seq2_mtp["total_vram_mb"] < USABLE_4GB_CARD_MB, (
        "n_seq_max=2 + mtp_enabled=True should be estimated comfortably under a 4GB "
        f"card's usable budget (matches the real successful load) — got {seq2_mtp['total_vram_mb']} MiB"
    )
