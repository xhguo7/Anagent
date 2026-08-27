#!/usr/bin/env python3
"""
Engine-layer unit tests (no server, GPU, or network required).
"""

import io
import os
import sys
import base64
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANAGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANAGENT_ROOT)
from PIL import Image

from anagent.engines.vllm import (
    _parse_ctx_overflow, _is_image_error, _strip_images, _shrink_text, VLLMEngine,
)
from anagent.engines.openai import OpenAIEngine


# ---------------------------------------------------------------- helpers ----
_TESTS = []
def test(fn):
    _TESTS.append(fn)
    return fn


def _png_bytes(size=(8, 8), color=(200, 30, 30)):
    """A tiny, valid PNG as raw bytes."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _engine():
    """An OpenAIEngine instance WITHOUT running __init__ (no API key needed); only the attributes the image helpers touch are set."""
    eng = OpenAIEngine.__new__(OpenAIEngine)
    eng.max_dimension = 2048
    return eng


def _text_len(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    n += len(p.get("text", "") or "")
    return n


def _num_images(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") in ("image", "image_url"))
    return n


# --------------------------------------------------------- vllm.py: parse ----
@test
def test_parse_ctx_overflow_matches():
    err = ("litellm.BadRequestError: OpenAIException - This model's maximum context "
           "length is 32768 tokens. However, you requested 551403 tokens")
    assert _parse_ctx_overflow(err) == (32768, 551403)


@test
def test_parse_ctx_overflow_request_has_variant():
    err = "maximum context length is 4096 tokens, but the request has 5000 input tokens"
    assert _parse_ctx_overflow(err) == (4096, 5000)


@test
def test_parse_ctx_overflow_none_on_unrelated():
    assert _parse_ctx_overflow("some other error") is None


# --------------------------------------------------- vllm.py: image error ----
@test
def test_is_image_error_true_on_real_error():
    err = ("litellm.BadRequestError: OpenAIException - cannot identify image file "
           "<_io.BytesIO object at 0x7f79235b80e0>")
    assert _is_image_error(err) is True


@test
def test_is_image_error_false_on_ctx_error():
    assert _is_image_error("maximum context length is 32768") is False


@test
def test_strip_images_removes_only_images():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image", "image": "/p.jpg"},
        ]},
    ]
    out, had = _strip_images(msgs)
    assert had is True
    assert _num_images(out) == 0
    # text and the string-content system message survive
    assert out[0]["content"] == "sys"
    assert any(p.get("type") == "text" for p in out[1]["content"])


@test
def test_strip_images_reports_no_image():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    _, had = _strip_images(msgs)
    assert had is False


# ---------------------------------------------- vllm.py: _shrink_text ---------
@test
def test_shrink_text_multiblock_meets_budget():
    # 5 large text blocks spread across messages, plus an image each
    msgs = [{"role": "system", "content": "S" * 2000}]
    for i in range(5):
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": (f"T{i} ") * 3000},               # ~15k chars
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]})
    before = _text_len(msgs)
    _, removed = _shrink_text(msgs, 40000)
    after = _text_len(msgs)
    assert removed == 40000, removed
    assert before - after >= 39000            # markers add a little back
    assert _num_images(msgs) == 5             # images NEVER touched
    assert any(m.get("content") == "S" * 2000 for m in msgs)  # small block kept intact


@test
def test_shrink_text_noop_on_nonpositive_budget():
    msgs = [{"role": "user", "content": "hello world"}]
    _, removed = _shrink_text(msgs, 0)
    assert removed == 0
    assert msgs[0]["content"] == "hello world"


@test
def test_shrink_text_zero_when_all_below_floor():
    # every block already at/under the min_keep floor -> nothing removable
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]}]
    _, removed = _shrink_text(msgs, 10000)
    assert removed == 0


@test
def test_shrink_text_single_huge_block():
    msgs = [{"role": "user", "content": "X" * 100000}]
    _, removed = _shrink_text(msgs, 60000)
    assert removed == 60000
    assert "[context truncated to fit the model window]" in msgs[0]["content"]


# -------------------------------------------- vllm.py: seed threading --------
def _seed_engine(seed):
    e = VLLMEngine.__new__(VLLMEngine)
    e.seed = seed
    e._seed_ctr = 0
    return e


@test
def test_next_seed_none_when_unseeded():
    assert _seed_engine(None)._next_seed() is None      # stochastic sampling


@test
def test_next_seed_distinct_and_reproducible():
    e = _seed_engine(5)
    seeds = [e._next_seed() for _ in range(3)]
    assert seeds == [5_000_000, 5_000_001, 5_000_002]   # distinct per call, deterministic


@test
def test_next_seed_different_base_no_overlap():
    # seed 5's call-range and seed 6's call-range never collide (< 1e6 calls/run)
    a = [_seed_engine(5)._next_seed() for _ in range(3)]
    b = [_seed_engine(6)._next_seed() for _ in range(3)]
    assert set(a).isdisjoint(set(b))


# -------------------------------------------- openai.py: image encoding ------
@test
def test_prepare_image_from_path(tmpfile="/tmp/_anagent_test_img.png"):
    Image.new("RGB", (10, 10), (0, 128, 255)).save(tmpfile, format="PNG")
    try:
        url = _engine()._prepare_image_content(tmpfile)
        assert url.startswith("data:image/png;base64,")
        # decodes back to a real image
        raw = base64.b64decode(url.split(",", 1)[1])
        Image.open(io.BytesIO(raw)).verify()
    finally:
        os.remove(tmpfile)


@test
def test_prepare_image_valid_bytes():
    url = _engine()._prepare_image_content(_png_bytes())
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    Image.open(io.BytesIO(raw)).verify()


@test
def test_prepare_image_invalid_bytes_raises():
    try:
        _engine()._prepare_image_content(b"this is not an image")
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-image bytes")


@test
def test_prepare_image_data_url_passthrough():
    du = "data:image/png;base64,AAAA"
    assert _engine()._prepare_image_content(du) == du


# ----------------------------------------- openai.py: format_messages --------
@test
def test_format_messages_raw_path_image_url_gets_encoded(tmpfile="/tmp/_anagent_test_img2.png"):
    Image.new("RGB", (10, 10), (10, 200, 10)).save(tmpfile, format="PNG")
    try:
        out = _engine().format_messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": tmpfile}},
            {"type": "text", "text": "hi"},
        ]}])
        url = out[0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), url[:40]
    finally:
        os.remove(tmpfile)


@test
def test_format_messages_http_and_data_url_passthrough():
    eng = _engine()
    for keep in ("http://x/y.png", "https://x/y.png", "data:image/png;base64,AAAA"):
        out = eng.format_messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": keep}},
        ]}])
        assert out[0]["content"][0]["image_url"]["url"] == keep


@test
def test_format_messages_image_type_to_data_url(tmpfile="/tmp/_anagent_test_img3.png"):
    Image.new("RGB", (10, 10), (5, 5, 5)).save(tmpfile, format="PNG")
    try:
        out = _engine().format_messages([{"role": "user", "content": [
            {"type": "image", "image": tmpfile},
            {"type": "text", "text": "hi"},
        ]}])
        assert out[0]["content"][0]["type"] == "image_url"
        assert out[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        os.remove(tmpfile)


# ------------------------------------------------------------------ main -----
def main():
    failed = 0
    print(f"Running {len(_TESTS)} engine-robustness tests\n")
    for fn in _TESTS:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
