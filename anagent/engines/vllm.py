"""
vLLM-served engine.
"""

import os
import re
import litellm

from utils.logger import setup_logger
logger = setup_logger(__name__)

from .engine import BaseEngine, EngineHub
from .openai import OpenAIEngine

# Max truncate-and-retry attempts on a context-overflow error
MAX_TRUNC_ATTEMPTS = 5


def _parse_ctx_overflow(err_str):
    """(max_ctx, requested_input) parsed from a context-length error, or None."""
    m = re.search(r"maximum context length is\s*(\d+).*?(?:request has|requested)\s*(\d+)",
                  err_str, re.I | re.S)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _is_image_error(err_str):
    """True if the server rejected the request because it couldn't decode an image."""
    s = err_str.lower()
    return ("cannot identify image file" in s or "invalid image" in s
            or "failed to load image" in s or "error while fetching image" in s)


def _strip_images(messages):
    """Drop every image part from the messages (keep the text). (messages, had_image?)."""
    had = False
    for msg in messages:
        c = msg.get("content")
        if isinstance(c, list):
            kept = []
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                    had = True
                    continue
                kept.append(part)
            msg["content"] = kept
    return messages, had


def _shrink_text(messages, remove_chars, min_keep=200):
    """Trim up to ``remove_chars`` characters of text spread across ALL text blocks."""
    if remove_chars <= 0:
        return messages, 0
    marker = "\n...[context truncated to fit the model window]...\n"

    # Index every text block once as (mi, pi, length)
    blocks = []
    for mi, msg in enumerate(messages):
        c = msg.get("content")
        if isinstance(c, str):
            blocks.append((mi, None, len(c)))
        elif isinstance(c, list):
            for pi, part in enumerate(c):
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text", "") or ""
                    blocks.append((mi, pi, len(t)))

    # Biggest blocks first: drain giant history / tool dumps, keep small important
    blocks.sort(key=lambda b: b[2], reverse=True)

    removed = 0
    for mi, pi, length in blocks:
        if removed >= remove_chars:
            break
        can_remove = length - min_keep          # keep a floor of head+tail per block
        if can_remove <= 0:
            continue
        take = min(can_remove, remove_chars - removed)
        t = messages[mi]["content"] if pi is None else messages[mi]["content"][pi]["text"]
        keep = length - take
        head = keep // 2
        newt = t[:head] + marker + t[length - (keep - head):]
        if pi is None:
            messages[mi]["content"] = newt
        else:
            messages[mi]["content"][pi]["text"] = newt
        removed += take
    return messages, removed


class VLLMEngine(OpenAIEngine):
    """Engine for a vLLM OpenAI-compatible server (local MLLM/LLM)."""

    def __init__(
        self,
        model_path: str,
        base_url: str = None,
        api_key: str = None,
        timeout: int = 600,
        max_dimension: int = 2048,
        seed: int = None,
        **kwargs,
    ):
        # Bypass OpenAIEngine.__init__ (it requires a real key + sets up cloud
        BaseEngine.__init__(self, model_path=model_path, **kwargs)

        self.base_url = base_url or os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if not self.base_url:
            raise ValueError(
                "VLLMEngine requires a base_url (e.g. http://localhost:8010/v1) via "
                "engine_kwargs['base_url'] or the VLLM_BASE_URL env var."
            )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.max_dimension = max_dimension
        # served name is everything after the 'vllm/' prefix
        self.served_model = model_path.split("vllm/", 1)[-1]
        # Sampling seed: each generate() call gets a distinct, reproducible seed
        self.seed = seed
        self._seed_ctr = 0

        logger.info(f"Initialized VLLM engine -> {self.base_url}  model={self.served_model}")
    def load_model(self):
        """No local model loading — served remotely by vLLM."""
        pass

    def _next_seed(self):
        """Distinct, reproducible seed per generate() call (base*1e6 + call index); None => stochastic sampling."""
        if self.seed is None:
            return None
        s = self.seed * 1_000_000 + self._seed_ctr
        self._seed_ctr += 1
        return s

    def generate(self, messages, temperature=None, max_tokens=None, repetition_penalty=None, **kwargs) -> str:
        temp = temperature if temperature is not None else self.temperature
        rep_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        formatted_messages = self.format_messages(messages)  # inherited: text + base64 image
        # One seed per logical generate() (stable across truncation retries below).
        call_seed = self._next_seed()
        seed_kw = {} if call_seed is None else {"seed": int(call_seed)}

        # Truncate-and-retry on a context-overflow error
        response = None
        for attempt in range(MAX_TRUNC_ATTEMPTS + 1):
            try:
                response = litellm.completion(
                    model=f"openai/{self.served_model}",   # openai/ + served name -> OpenAI-compatible route
                    messages=formatted_messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=self.top_p,
                    repetition_penalty=rep_pen,
                    api_key=self.api_key,
                    api_base=self.base_url,
                    timeout=self.timeout,
                    **seed_kw,
                )
                break
            except Exception as e:
                es = str(e)
                # An undecodable image: drop the image(s) and retry text-only
                if _is_image_error(es) and attempt < MAX_TRUNC_ATTEMPTS:
                    formatted_messages, had_image = _strip_images(formatted_messages)
                    if had_image:
                        logger.warning(f"[VLLM] server could not decode an image; retrying WITHOUT "
                                       f"the image, text-only ({attempt + 1}/{MAX_TRUNC_ATTEMPTS})")
                        continue
                info = _parse_ctx_overflow(es)
                if info is None or attempt >= MAX_TRUNC_ATTEMPTS:
                    logger.error(f"[VLLM] completion failed @ {self.base_url} model={self.served_model}: "
                                 f"{type(e).__name__}: {e}")
                    raise
                max_ctx, req = info
                # input must fit in (max_ctx - output budget); trim the overflow + margin
                remove_tokens = (req - (max_ctx - max_tok)) + 512
                # Trim across ALL text blocks (not just the longest) so an overflow
                formatted_messages, removed = _shrink_text(formatted_messages, int(remove_tokens * 4))
                if removed <= 0:
                    # Every text block is already at its floor
                    formatted_messages, had_image = _strip_images(formatted_messages)
                    if not had_image:
                        logger.error(f"[VLLM] input {req} > ctx {max_ctx} and no trimmable text "
                                     f"or image remains; failing. Raise the server --max-model-len.")
                        raise
                    logger.warning(f"[VLLM] input {req} > ctx {max_ctx}: no trimmable text left; "
                                   f"dropped image(s), retrying text-only ({attempt + 1}/{MAX_TRUNC_ATTEMPTS})")
                    continue
                logger.warning(f"[VLLM] input {req} + {max_tok} output tok > ctx {max_ctx}: trimmed "
                               f"~{removed // 4} tokens across text blocks, retrying "
                               f"({attempt + 1}/{MAX_TRUNC_ATTEMPTS})")

        model_response = response.choices[0].message.content
        return model_response.strip() if model_response else ""


# Register under the 'vllm' key; EngineHub routes model paths starting with 'vllm/'.
EngineHub.register_engine("vllm", VLLMEngine)
