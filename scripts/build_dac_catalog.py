#!/usr/bin/env python3
"""Regenerate the vendored OCI Dedicated AI Cluster catalog.

The Sizing tab has to work offline, so it reads a committed JSON catalog rather
than calling Oracle or Hugging Face at request time. This script is the only
thing that touches the network, and it is run by hand (`make dac-catalog`) when
Oracle publishes new validated models.

Two sources are joined:

* Oracle's "Compatible <family> Models" pages give the authoritative triple
  (Hugging Face id, model capability, validated unit shapes). Only Oracle can
  say which shapes are validated, so this half is copied, never inferred.
* Hugging Face gives the architecture that the KV-cache and roofline math needs
  (layers, heads, KV heads, head dim, sliding window, MoE expert layout). The
  parameter count comes from the safetensors index rather than the config,
  because configs do not carry it and estimating it from dimensions drifts on
  tied embeddings and MoE.

Models Oracle lists but Hugging Face cannot describe (gated repos, missing
config) are still written out with `architecture: null`. Dropping them would
silently shrink the catalog Oracle says is validated; keeping them lets the app
show the model, its validated shapes, and an honest "no architecture data" state
instead of pretending the model does not exist.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html import unescape
from pathlib import Path
from typing import Any

DOCS_ROOT = "https://docs.oracle.com/en-us/iaas/Content/generative-ai/"
INDEX_PAGE = DOCS_ROOT + "imported-models.htm"
HF_API = "https://huggingface.co/api/models/"
HF_RAW = "https://huggingface.co/{model_id}/resolve/main/config.json"

# Oracle's index page does not link every family page it publishes; DeepSeek has
# a live page that nothing on imported-models.htm points at. Listed explicitly so
# the catalog does not silently lose a family whenever Oracle's nav drifts.
EXTRA_FAMILY_KEYS = ("deepseek",)

# Meta, Google and Microsoft gate their repos behind a click-through licence, so
# config.json 401s for an anonymous reader. Community mirrors republish the same
# config byte-for-byte, so architecture is read from the mirror while the
# parameter count still comes from the canonical repo's safetensors index. Each
# attempt is recorded in `architecture_source` so the substitution is visible.
MIRROR_PREFIXES = ("unsloth/", "NousResearch/")
# Oracle's docs CDN rejects urllib's default signature, so requests are sent
# with an ordinary browser header set. Nothing here is authentication.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "api"
    / "src"
    / "waqil_api"
    / "data"
    / "dac"
)
MODELS_OUTPUT = DATA_DIR / "models.json"
BENCHMARKS_OUTPUT = DATA_DIR / "benchmarks.json"

PERFORMANCE_PAGE = DOCS_ROOT + "performance.htm"
# Oracle links the Cohere and Meta benchmark hubs from performance.htm but not
# the OpenAI one, so it is named here. Only the gpt-oss pages state which GPUs
# their cluster unit actually is; every other family reports an opaque marketing
# unit ("one Large Generic unit"), which is why calibration can only anchor on
# these two. The rest are still scraped and stored as measured reference rows.
BENCHMARK_HUBS = ("benchmarks-cohere", "benchmarks-meta", "benchmarks-openai")

# Benchmarks are published under OCI service model ids, which do not match the
# Hugging Face ids the catalog is keyed by. Only the hardware-anchored grids need
# this join (calibration has to read the model's architecture), so the map is
# written out explicitly rather than guessed from the id shape — `meta.llama-3.3
# -70b-instruct` and `meta-llama/Llama-3.3-70B-Instruct` have no mechanical
# relationship, and a wrong join would calibrate against the wrong dimensions.
SERVICE_MODEL_TO_HF = {
    "openai.gpt-oss-120b": "openai/gpt-oss-120b",
    "openai.gpt-oss-20b": "openai/gpt-oss-20b",
}

# The seven columns Oracle publishes, in order, mapped to stable field names.
BENCHMARK_FIELDS = (
    "concurrency",
    "ttft_s",
    "inference_speed_tps",
    "token_throughput_tps",
    "request_latency_s",
    "request_throughput",
    "total_throughput_tps",
)

# Oracle writes the same shape both ways across pages (A100_80GB_X2 vs
# A100_80G_X2). shapes.json owns the canonical spelling; this mirrors its
# alias table so the catalog only ever emits canonical ids.
SHAPE_ALIASES = {
    "A100_80GB_X1": "A100_80G_X1",
    "A100_80GB_X2": "A100_80G_X2",
    "A100_80GB_X4": "A100_80G_X4",
    "A100_80GB_X8": "A100_80G_X8",
    "A100_40GB_X1": "A100_40G_X1",
    "A100_40GB_X2": "A100_40G_X2",
    "A100_40GB_X4": "A100_40G_X4",
    "A100_40GB_X8": "A100_40G_X8",
}

FAMILY_LABELS = {
    "alibaba": "Alibaba Qwen",
    "deepseek": "DeepSeek",
    "google": "Google Gemma",
    "meta": "Meta Llama",
    "microsoft": "Microsoft Phi",
    "minimax": "MiniMax",
    "mistral": "Mistral",
    "moonshot-ai": "Moonshot AI Kimi",
    "nvidia": "NVIDIA Nemotron",
    "openai": "OpenAI",
    "openai-oss": "OpenAI GptOss",
    "zai": "Z.ai GLM",
}


def fetch(url: str, *, retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):  # gated or absent: no retry helps
                raise
            last = error
        except Exception as error:  # noqa: BLE001 - transient network
            last = error
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def strip_tags(fragment: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def family_pages() -> list[tuple[str, str]]:
    """(family_key, url) for every "Compatible <family> Models" page."""
    index = fetch(INDEX_PAGE)
    keys = {key for _, key in re.findall(r'href="(imported-([a-z0-9\-]+)-models\.htm)"', index)}
    keys.update(EXTRA_FAMILY_KEYS)
    return [(key, f"{DOCS_ROOT}imported-{key}-models.htm") for key in sorted(keys)]


def parse_family_page(html: str) -> list[dict[str, Any]]:
    """Every (hugging face id, capability, validated shapes) row on the page.

    Oracle renders one table per model generation with a fixed three-column
    layout, so rows are matched positionally. Shapes appear either as a plain
    string or as a bullet list, hence the split on both commas and list items.
    """
    rows: list[dict[str, Any]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S | re.I)
        if len(cells) < 3:
            continue
        link = re.search(r'href="https://huggingface\.co/([^"#?]+)"', cells[0])
        if not link:
            continue
        model_id = link.group(1).strip().rstrip("/")
        capability = strip_tags(cells[1]).strip().upper()
        shapes: list[str] = []
        for token in re.split(r"</li>|,|<br\s*/?>", cells[2], flags=re.I):
            name = strip_tags(token).strip().upper().replace(" ", "")
            if re.fullmatch(r"[A-Z0-9_]+_X\d+", name):
                shapes.append(SHAPE_ALIASES.get(name, name))
        rows.append(
            {
                "id": model_id,
                "capability": capability,
                "validated_shapes": list(dict.fromkeys(shapes)),
            }
        )
    return rows


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal repos nest the language model under `text_config`.

    The vision tower has its own (much smaller) dimensions, and reading those by
    mistake would understate the KV cache by an order of magnitude.
    """
    nested = config.get("text_config")
    if isinstance(nested, dict) and "num_hidden_layers" in nested:
        merged = {**config, **nested}
        return merged
    return config


def _sliding_window(config: dict[str, Any]) -> int | None:
    window = config.get("sliding_window")
    if not isinstance(window, int) or window <= 0:
        return None
    # Qwen ships `sliding_window` alongside `use_sliding_window: false`; the
    # window is inert in that case and must not shrink the cache estimate.
    if config.get("use_sliding_window") is False:
        return None
    return window


def _sliding_window_ratio(config: dict[str, Any]) -> float | None:
    """Fraction of layers that actually use the sliding window.

    Gemma-style interleaving (5 local : 1 global) means a flat "all layers are
    windowed" assumption undercounts the cache badly on long context.
    """
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        local = sum(1 for item in layer_types if isinstance(item, str) and "sliding" in item)
        return local / len(layer_types)
    pattern = config.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        return (pattern - 1) / pattern
    return None


def _moe(config: dict[str, Any]) -> dict[str, Any] | None:
    experts = (
        config.get("n_routed_experts")
        or config.get("num_local_experts")
        or config.get("num_experts")
    )
    per_token = config.get("num_experts_per_tok") or config.get("num_experts_per_token")
    if not isinstance(experts, int) or not isinstance(per_token, int):
        return None
    if experts <= 1:
        return None
    shared = config.get("n_shared_experts")
    shared_intermediate = config.get("shared_expert_intermediate_size")
    if not isinstance(shared, int):
        shared = 1 if isinstance(shared_intermediate, int) and shared_intermediate else 0
    return {
        "num_experts": experts,
        "experts_per_token": per_token,
        "expert_intermediate_size": config.get("moe_intermediate_size")
        or config.get("intermediate_size"),
        "num_shared_experts": shared,
        "shared_expert_intermediate_size": shared_intermediate,
    }


def _expert_params(config: dict[str, Any], moe: dict | None) -> tuple[int, int]:
    """(all expert parameters, parameters routed per token).

    Kept separate from the totals because expert weights are frequently stored
    at a different precision from the rest of the model — gpt-oss quantizes its
    experts to MXFP4 while attention, router and embeddings stay BF16. Treating
    the model as uniformly BF16 overstates the bytes read per token by ~2x and
    shows up as a bogus 'memory bandwidth utilization above 70%' in the fit.
    """
    if not moe:
        return 0, 0
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    expert_dim = moe.get("expert_intermediate_size")
    if not all(isinstance(value, int) and value > 0 for value in (layers, hidden, expert_dim)):
        return 0, 0
    per_expert = 3 * hidden * expert_dim
    total = moe["num_experts"] * per_expert * layers
    active = moe["experts_per_token"] * per_expert * layers
    return int(total), int(active)


def _weight_dtypes(config: dict[str, Any]) -> tuple[str, str]:
    """(dense dtype, expert dtype) as stored on disk.

    `quantization_config.modules_to_not_convert` is what says the attention and
    embedding tensors keep full precision while the experts do not.
    """
    dense = config.get("torch_dtype") or config.get("dtype") or "bf16"
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return dense, dense
    method = str(quantization.get("quant_method") or "").lower()
    if not method:
        return dense, dense
    skipped = " ".join(str(item) for item in quantization.get("modules_to_not_convert") or [])
    # If attention is excluded from quantization, only the experts are compressed.
    if "self_attn" in skipped or "embed_tokens" in skipped:
        return dense, method
    return method, method


def _active_params(total: int | None, config: dict[str, Any], moe: dict | None) -> int | None:
    """Parameters touched per token — the number that drives decode bandwidth.

    For a dense model this is the total. For MoE the router reads only
    `experts_per_token` experts, so the rest occupy HBM without being read.

    This is built up analytically (embeddings + attention + router + shared
    experts + the active experts) rather than by subtracting inactive experts
    from the safetensors total. Subtraction looks simpler but inherits any
    distortion in the reported total: gpt-oss ships MXFP4 expert blocks whose
    element count is not the logical parameter count, and subtracting there
    overstates active parameters by ~80%. The analytic sum reproduces the
    published 5.1B for gpt-oss-120b and 3B for Qwen3-30B-A3B.
    """
    if not moe:
        return total
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or (
        hidden // heads if isinstance(hidden, int) and isinstance(heads, int) and heads else None
    )
    vocab = config.get("vocab_size")
    expert_dim = moe.get("expert_intermediate_size")
    required = (layers, hidden, heads, kv_heads, head_dim, vocab, expert_dim)
    if not all(isinstance(value, int) and value > 0 for value in required):
        return total

    embedding = vocab * hidden
    if not config.get("tie_word_embeddings", False):
        embedding *= 2
    attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    router = hidden * moe["num_experts"]
    shared_dim = moe.get("shared_expert_intermediate_size")
    shared = (
        3 * hidden * shared_dim * max(1, moe.get("num_shared_experts") or 0)
        if isinstance(shared_dim, int) and shared_dim > 0
        else 0
    )
    active_experts = moe["experts_per_token"] * 3 * hidden * expert_dim
    per_layer = attention + router + shared + active_experts
    active = embedding + layers * per_layer
    if total is not None:
        active = min(active, total)
    return int(active)


def _attention_type(config: dict[str, Any]) -> str:
    if config.get("kv_lora_rank"):
        return "mla"
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    if isinstance(heads, int) and isinstance(kv_heads, int) and kv_heads < heads:
        return "gqa"
    return "mha"


def _load_config(model_id: str) -> tuple[dict[str, Any], str] | None:
    """The repo's config.json, falling back to a community mirror if gated."""
    basename = model_id.split("/", 1)[-1]
    candidates = [model_id]
    candidates += [prefix + basename for prefix in MIRROR_PREFIXES]
    candidates += ["NousResearch/Meta-" + basename]
    for candidate in dict.fromkeys(candidates):
        try:
            raw = json.loads(fetch(HF_RAW.format(model_id=candidate), retries=1))
        except Exception:  # noqa: BLE001 - gated/absent repos are expected
            continue
        if isinstance(raw, dict) and raw.get("num_hidden_layers") or (
            isinstance(raw, dict) and isinstance(raw.get("text_config"), dict)
        ):
            return raw, candidate
    return None


def architecture_for(model_id: str) -> dict[str, Any] | None:
    """Normalized architecture for one Hugging Face repo, or None if unreadable."""
    loaded = _load_config(model_id)
    if loaded is None:
        return None
    raw, config_source = loaded
    config = _text_config(raw)
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    if not all(isinstance(value, int) and value > 0 for value in (layers, hidden, heads)):
        return None
    kv_heads = config.get("num_key_value_heads")
    if not isinstance(kv_heads, int) or kv_heads <= 0:
        kv_heads = heads
    head_dim = config.get("head_dim")
    if not isinstance(head_dim, int) or head_dim <= 0:
        head_dim = hidden // heads

    total_params: int | None = None
    try:
        meta = json.loads(fetch(HF_API + model_id))
        safetensors = meta.get("safetensors") or {}
        candidate = safetensors.get("total")
        if isinstance(candidate, int) and candidate > 0:
            total_params = candidate
    except Exception:  # noqa: BLE001 - param count is optional, not fatal
        total_params = None

    moe = _moe(config)
    expert_total, expert_active = _expert_params(config, moe)
    dense_dtype, expert_dtype = _weight_dtypes(config)
    architecture: dict[str, Any] = {
        "config_source": config_source,
        "params_total": total_params,
        "params_active": _active_params(total_params, config, moe),
        "params_expert_total": expert_total,
        "params_expert_active": expert_active,
        "weight_dtype": dense_dtype,
        "expert_dtype": expert_dtype,
        "num_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "attention_type": _attention_type(config),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "sliding_window": _sliding_window(config),
        "sliding_window_ratio": _sliding_window_ratio(config),
        "rope_theta": config.get("rope_theta"),
        "torch_dtype": dense_dtype,
        "model_type": config.get("model_type"),
        "is_moe": bool(moe),
        "moe": moe,
    }
    if architecture["attention_type"] == "mla":
        architecture["mla"] = {
            "kv_lora_rank": config.get("kv_lora_rank"),
            "qk_rope_head_dim": config.get("qk_rope_head_dim"),
            "qk_nope_head_dim": config.get("qk_nope_head_dim"),
            "v_head_dim": config.get("v_head_dim"),
        }
    return architecture


def benchmark_pages() -> list[str]:
    """Every per-model benchmark page URL, gathered from the family hubs."""
    pages: set[str] = set()
    for hub in BENCHMARK_HUBS:
        try:
            html_text = fetch(f"{DOCS_ROOT}{hub}.htm")
        except Exception as error:  # noqa: BLE001 - one missing hub is survivable
            print(f"  ! {hub}: {error}", file=sys.stderr)
            continue
        # Hubs link some pages bare and others with an #anchor fragment, so the
        # fragment is matched and discarded rather than assumed absent. Missing
        # it silently drops the only two hardware-anchored grids Oracle has.
        pages.update(re.findall(r'href="(benchmark-[a-z0-9\-\.]*\.htm)(?:#[^"]*)?"', html_text))
    return sorted(DOCS_ROOT + page for page in pages)


def _scenario_spec(summary: str) -> dict[str, Any]:
    """Prompt/response token counts parsed from a table's summary attribute.

    The summary is used rather than the prose above the table because it is
    attached to the exact grid whose numbers are being read, so a page that
    reorders its sections cannot mislabel a scenario.

    Oracle phrases the spec three different ways across its benchmark pages, so
    all three are matched explicitly rather than approximated by one loose
    pattern. Guessing here would silently mis-scale the calibration: reading
    "128,000 tokens" as 128 would make the model look wildly wrong on long
    context and get compensated for elsewhere in the fit.
    """
    text = " ".join(summary.split())
    spec: dict[str, Any] = {"summary": text}

    def number(raw: str) -> int:
        return int(raw.replace(",", ""))

    # 1. "scenario: normal distribution of N(480,240), and response length:
    #     normal distribution of N(300,150)."
    distributions = re.findall(r"N\(\s*([\d,]+)\s*,\s*([\d,]+)\s*\)", text)
    if len(distributions) >= 2:
        spec["prompt_tokens"], spec["prompt_stddev"] = map(number, distributions[0])
        spec["response_tokens"], spec["response_stddev"] = map(number, distributions[1])
        return spec

    # 2. "scenario of prompt length of 2,000 tokens, and response length of 200 tokens."
    prompt = re.search(r"prompt length of\s*([\d,]+)\s*tokens?", text, re.I)
    response = re.search(r"response length of\s*([\d,]+)\s*tokens?", text, re.I)
    if prompt and response:
        spec["prompt_tokens"] = number(prompt.group(1))
        spec["response_tokens"] = number(response.group(1))
        spec["prompt_stddev"] = spec["response_stddev"] = 0
        return spec

    # 3. "scenario with prompt and response length of 100 tokens each."
    both = re.search(r"prompt and response length of\s*([\d,]+)\s*tokens?", text, re.I)
    if both:
        spec["prompt_tokens"] = spec["response_tokens"] = number(both.group(1))
        spec["prompt_stddev"] = spec["response_stddev"] = 0
    return spec


def parse_benchmark_page(html_text: str, url: str) -> list[dict[str, Any]]:
    """Every benchmark grid on one page, one record per (scenario, unit)."""
    headings = [
        " ".join(unescape(re.sub(r"<[^>]+>", "", match)).split())
        for match in re.findall(r"<h2[^>]*>(.*?)</h2>", html_text, flags=re.S)
    ]
    grids: list[dict[str, Any]] = []
    for index, table in enumerate(re.findall(r"<table[^>]*>.*?</table>", html_text, flags=re.S)):
        summary_match = re.search(r'summary="([^"]*)"', table)
        # Oracle wraps the summary across source lines, so it is collapsed to a
        # single spaced string before anything is matched against it.
        summary = " ".join(unescape(summary_match.group(1)).split()) if summary_match else ""
        headers = [strip_tags(cell) for cell in re.findall(r"<th[^>]*>(.*?)</th>", table, flags=re.S)]
        if not headers or "Concurrency" not in headers[0]:
            continue  # not a benchmark grid (region tables share the page)

        rows: list[dict[str, Any]] = []
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S):
            cells = [strip_tags(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)]
            if len(cells) != len(BENCHMARK_FIELDS):
                continue
            try:
                values = [float(cell.replace(",", "")) for cell in cells]
            except ValueError:
                continue
            row = dict(zip(BENCHMARK_FIELDS, values, strict=True))
            row["concurrency"] = int(row["concurrency"])
            rows.append(row)
        if not rows:
            continue

        # Oracle reports request throughput per minute on some pages and per
        # second on others. Normalizing here means the calibration never has to
        # guess which unit a stored row is in.
        per_minute = bool(re.search(r"per minute|RPM", " ".join(headers), re.I))
        for row in rows:
            row["request_throughput_rps"] = (
                row.pop("request_throughput") / 60.0 if per_minute else row.pop("request_throughput")
            )

        shape = re.search(r"\b((?:OAI_)?(?:A10|A100_40G|A100_80G|H100|H200|B200)_X\d+)\b", summary)
        unit = re.search(r"hosted on (\w+) ([A-Za-z0-9_ ]+?) unit", summary)
        model = re.search(r"benchmarks with the ([\w\.\-]+)\s*\(", summary)
        grids.append(
            {
                "source_url": url,
                "scenario": headings[index] if index < len(headings) else f"scenario-{index + 1}",
                "model": model.group(1) if model else None,
                "hf_id": SERVICE_MODEL_TO_HF.get(model.group(1)) if model else None,
                "shape": shape.group(1) if shape else None,
                "unit_label": unit.group(2).strip() if unit else None,
                "hardware_known": bool(shape),
                **_scenario_spec(summary),
                "rows": rows,
            }
        )
    return grids


def build_benchmarks(workers: int) -> dict[str, Any]:
    pages = benchmark_pages()
    print(f"found {len(pages)} benchmark pages", file=sys.stderr)

    def load(url: str) -> list[dict[str, Any]]:
        try:
            return parse_benchmark_page(fetch(url), url)
        except Exception as error:  # noqa: BLE001 - skip a page, keep the rest
            print(f"  ! {url}: {error}", file=sys.stderr)
            return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        grids = [grid for page in pool.map(load, pages) for grid in page]

    anchored = [grid for grid in grids if grid["hardware_known"]]
    rows = sum(len(grid["rows"]) for grid in grids)
    print(
        f"parsed {len(grids)} grids / {rows} rows "
        f"({len(anchored)} grids on named hardware)",
        file=sys.stderr,
    )
    return {
        "generated_at": date.today().isoformat(),
        "source_urls": [PERFORMANCE_PAGE] + [f"{DOCS_ROOT}{hub}.htm" for hub in BENCHMARK_HUBS],
        "notes": [
            "Every row is copied from Oracle's published benchmark tables.",
            "hardware_known marks grids whose cluster unit Oracle names as a GPU shape.",
            "Only those can calibrate the roofline model; the rest report an opaque",
            "unit label ('Large Generic'), so their GPU count and type are unknown.",
            "request_throughput_rps is normalized to per-second on every row.",
        ],
        "grid_count": len(grids),
        "row_count": rows,
        "calibration_grid_count": len(anchored),
        "grids": grids,
    }


def build(limit: int | None, workers: int) -> dict[str, Any]:
    pages = family_pages()
    print(f"found {len(pages)} family pages", file=sys.stderr)

    records: dict[str, dict[str, Any]] = {}
    source_urls = [INDEX_PAGE]
    for key, url in pages:
        source_urls.append(url)
        try:
            rows = parse_family_page(fetch(url))
        except Exception as error:  # noqa: BLE001 - one bad page must not abort
            print(f"  ! {key}: {error}", file=sys.stderr)
            continue
        print(f"  {key}: {len(rows)} models", file=sys.stderr)
        for row in rows:
            existing = records.get(row["id"])
            if existing:  # a model listed on two pages keeps the union of shapes
                existing["validated_shapes"] = list(
                    dict.fromkeys(existing["validated_shapes"] + row["validated_shapes"])
                )
                continue
            records[row["id"]] = {
                **row,
                "family": FAMILY_LABELS.get(key, key.title()),
                "family_key": key,
            }

    ordered = sorted(records.values(), key=lambda item: item["id"].lower())
    if limit:
        ordered = ordered[:limit]
    print(f"resolving architecture for {len(ordered)} models", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        architectures = list(pool.map(lambda item: architecture_for(item["id"]), ordered))

    resolved = 0
    for record, architecture in zip(ordered, architectures, strict=True):
        record["architecture"] = architecture
        if architecture:
            resolved += 1
        else:
            print(f"  ? no architecture: {record['id']}", file=sys.stderr)
    print(f"resolved {resolved}/{len(ordered)}", file=sys.stderr)

    return {
        "generated_at": date.today().isoformat(),
        "source_urls": source_urls,
        "notes": [
            "validated_shapes is copied verbatim from Oracle's compatible-models pages.",
            "It is the recommended shape, which is often larger than the memory minimum.",
            "architecture is null when the Hugging Face repo is gated or has no config.json.",
        ],
        "model_count": len(ordered),
        "architecture_count": resolved,
        "models": ordered,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N models")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models-output", type=Path, default=MODELS_OUTPUT)
    parser.add_argument("--benchmarks-output", type=Path, default=BENCHMARKS_OUTPUT)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    args = parser.parse_args()

    if not args.skip_models:
        _write(args.models_output, build(args.limit, args.workers))
    if not args.skip_benchmarks:
        _write(args.benchmarks_output, build_benchmarks(args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
