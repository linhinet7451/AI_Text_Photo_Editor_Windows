import os

# 啟動時 transformers 會噴「requires torchvision (not installed); falling back」警告
# （無害但嚇人）：在 import diffusers/transformers 之前先把 verbosity 壓到 error，
# 並以 warnings 過濾該訊息樣式。不為此新增 torchvision 依賴。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings

warnings.filterwarnings(
    "ignore", message=r".*requires torchvision \(not installed\).*"
)
warnings.filterwarnings("ignore", message=r".*torchvision.*falling back.*")

import inspect
import io
import json
import logging
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting

try:
    from diffusers import DPMSolverMultistepScheduler
except Exception:  # 舊版 diffusers 或測試替身缺少此類別時，略過 scheduler 替換
    DPMSolverMultistepScheduler = None

logging.getLogger("transformers").setLevel(logging.ERROR)

APP_TITLE = "AI 文字修圖工具（本機版）"
DEFAULT_MODEL_ID = os.environ.get(
    "MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"
)
DEFAULT_PROC_SIDE = int(os.environ.get("MAX_SIDE", "512"))
TRANSLATE_MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
QUALITY_SUFFIX = "high quality, detailed, photorealistic"
CJK_THRESHOLD = 0.3
FULL_MASK_COVERAGE = 0.98
EDITOR_MIN_MASK_PIXELS = 16
VIDEO_MAX_SECONDS = 65

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

DEFAULT_WORKER_URL = "http://127.0.0.1:8001"
BACKEND_CHOICES = [("本機（自己的電腦）", "local"), ("雲端 GPU（worker）", "cloud")]

MODEL_HINT = {
    DEFAULT_MODEL_ID: "通用修圖／換衣／背景替換（預設，SD 1.5 Inpainting）",
}

# ---------------------------------------------------------------------------
# 設定持久化（settings.json，與 app.py 同層；不進版控）
# ---------------------------------------------------------------------------
def default_settings():
    return {
        "backend": "local",
        "worker_url": DEFAULT_WORKER_URL,
        "worker_token": "",
        "instance_id": "",
    }


def _clean_settings(data):
    s = default_settings()
    if isinstance(data, dict):
        for key in s:
            if isinstance(data.get(key), str):
                s[key] = data[key]
    if s["backend"] not in ("local", "cloud"):
        s["backend"] = "local"
    return s


def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return _clean_settings(json.load(f))
    except Exception:
        return default_settings()


def save_settings(settings):
    s = _clean_settings(settings)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    return s


def backend_status_text(backend) -> str:
    if backend == "cloud":
        return "目前運算位置：雲端 GPU（請在「雲端設定」測試連線）"
    return "目前運算位置：本機（自己的電腦）"


# ---------------------------------------------------------------------------
# 模型狀態（供 UI 顯示）
# ---------------------------------------------------------------------------
_pipe_status = {"state": "未載入", "detail": ""}
_pipe_status_lock = threading.Lock()


def _set_pipe_status(state: str, detail: str = ""):
    with _pipe_status_lock:
        _pipe_status["state"] = state
        _pipe_status["detail"] = detail


def _get_pipe_status_text() -> str:
    with _pipe_status_lock:
        state = _pipe_status["state"]
        detail = _pipe_status["detail"]
    return f"模型狀態：{state}" + (f"（{detail}）" if detail else "")


# ---------------------------------------------------------------------------
# 基本工具
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_size(width: int, height: int, max_side: int = DEFAULT_PROC_SIDE):
    """等比縮小到長邊 <= max_side（不放大），並對齊 8 的倍數、最小 64。"""
    scale = min(max_side / max(width, height), 1.0)
    w = max(64, int(width * scale))
    h = max(64, int(height * scale))
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


def fit_long_side(width: int, height: int, target: int):
    """等比縮放使長邊 == target（可放大或縮小），對齊 8 的倍數、最小 64。"""
    scale = target / max(width, height)
    w = max(64, int(round(width * scale)))
    h = max(64, int(round(height * scale)))
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


def pil_from_any(image_like) -> Optional[Image.Image]:
    if image_like is None:
        return None
    if isinstance(image_like, Image.Image):
        return image_like
    if isinstance(image_like, np.ndarray):
        arr = image_like
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(image_like, str) and os.path.exists(image_like):
        return Image.open(image_like)
    return None


# ---------------------------------------------------------------------------
# 中文 → 英文翻譯層（P0-1）
# SD 1.5 的文字編碼器（CLIP）只懂英文，中文指示先翻成英文再送模型。
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_SEGMENT_SPLIT_RE = re.compile(r"(?<=[。！？；;.!?\n])")

_translator_cache = {"tok": None, "model": None, "failed": False}
_translator_lock = threading.Lock()


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _CJK_RE.match(c)) / len(chars)


def needs_translation(text: str) -> bool:
    return bool(text) and cjk_ratio(text) >= CJK_THRESHOLD


def split_segments(text: str):
    return [p.strip() for p in _SEGMENT_SPLIT_RE.split(text or "") if p and p.strip()]


def _load_translator():
    """Lazy load Helsinki-NLP/opus-mt-zh-en，模組級快取；失敗會記住並 raise。"""
    if _translator_cache["model"] is not None:
        return _translator_cache["tok"], _translator_cache["model"]
    if _translator_cache["failed"]:
        raise RuntimeError("翻譯模型先前載入失敗")
    with _translator_lock:
        if _translator_cache["model"] is not None:
            return _translator_cache["tok"], _translator_cache["model"]
        if _translator_cache["failed"]:
            raise RuntimeError("翻譯模型先前載入失敗")
        try:
            from transformers import MarianMTModel, MarianTokenizer

            tok = MarianTokenizer.from_pretrained(TRANSLATE_MODEL_ID)
            model = MarianMTModel.from_pretrained(TRANSLATE_MODEL_ID)
            model.eval()
        except Exception:
            _translator_cache["failed"] = True
            raise
        _translator_cache["tok"] = tok
        _translator_cache["model"] = model
        return tok, model


def _translate_segment(tok, model, segment: str) -> str:
    batch = tok(
        [segment], return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        generated = model.generate(**batch, max_length=512)
    return tok.decode(generated[0], skip_special_tokens=True).strip()


def translate_text(text: str):
    """回傳 (文字, 狀態)，狀態為 'ok' / 'not_needed' / 'unavailable'。

    含中文比例夠高才翻；逐句翻譯，句中無中文的片段原樣保留。
    翻譯模型載入或翻譯失敗時回傳原文並標記 'unavailable'。
    """
    text = text or ""
    if not needs_translation(text):
        return text, "not_needed"
    try:
        tok, model = _load_translator()
        out_parts = []
        for seg in split_segments(text):
            if _CJK_RE.search(seg):
                out_parts.append(_translate_segment(tok, model, seg))
            else:
                out_parts.append(seg)
        return " ".join(p for p in out_parts if p), "ok"
    except Exception:
        return text, "unavailable"


def assemble_prompt(edit_text: str, preserve_text: str) -> str:
    parts = []
    if edit_text and edit_text.strip():
        parts.append(edit_text.strip().rstrip(","))
    if preserve_text and preserve_text.strip():
        # 保留描述直接作為描述句接在指示後（模型看得懂的自然英文句）。
        parts.append(preserve_text.strip().rstrip(","))
    parts.append(QUALITY_SUFFIX)
    return ", ".join(p for p in parts if p)


def build_prompts(edit_instruction, preserve_instruction, negative_prompt):
    edit_raw = (edit_instruction or "").strip()
    preserve_raw = (preserve_instruction or "").strip()
    negative_raw = (negative_prompt or "").strip()

    edit_en, st1 = translate_text(edit_raw)
    preserve_en, st2 = translate_text(preserve_raw)
    negative_en, st3 = translate_text(negative_raw)

    statuses = (st1, st2, st3)
    if "unavailable" in statuses:
        translation = "unavailable"
    elif "ok" in statuses:
        translation = "ok"
    else:
        translation = "not_needed"

    return {
        "prompt": assemble_prompt(edit_en, preserve_en),
        "prompt_original": assemble_prompt(edit_raw, preserve_raw),
        "negative_prompt": negative_en,
        "negative_prompt_original": negative_raw,
        "translation": translation,
    }


# ---------------------------------------------------------------------------
# 遮罩處理與合成回貼（P0-2）
# ---------------------------------------------------------------------------
def binarize_mask(mask_img: Image.Image) -> Image.Image:
    m = mask_img.convert("L")
    return m.point(lambda x: 255 if x >= 128 else 0)


def mask_coverage(mask_bin: Image.Image) -> float:
    arr = np.asarray(mask_bin)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).mean())


def mask_from_editor(editor_value, size=None) -> Optional[Image.Image]:
    """從 gr.ImageEditor 的值（dict: background/layers/composite）抽出遮罩。

    layers 為保留 alpha 的 RGBA 圖層；筆刷固定白色，
    遮罩值 = 各圖層 alpha × 亮度的最大值，再二值化。
    沒有任何有效塗鴉時回傳 None。
    """
    if not isinstance(editor_value, dict):
        return None
    layers = editor_value.get("layers") or []
    acc = None
    for layer in layers:
        img = pil_from_any(layer)
        if img is None:
            continue
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1]).astype(np.float32)
        lum = np.asarray(rgba.convert("L")).astype(np.float32)
        contrib = alpha * lum / 255.0
        acc = contrib if acc is None else np.maximum(acc, contrib)
    if acc is None:
        return None
    mask = binarize_mask(Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), "L"))
    if int((np.asarray(mask) > 0).sum()) < EDITOR_MIN_MASK_PIXELS:
        return None
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def compute_crop_box(mask_bin: Image.Image, pad_ratio: float = 0.25, min_side: int = 64):
    """遮罩 bbox 加 25% 上下文 padding；裁切邊至少 min_side；夾在圖內。"""
    bbox = mask_bin.getbbox()
    if bbox is None:
        return None
    width, height = mask_bin.size
    x0, y0, x1, y1 = bbox
    pad_x = max(1, int(round((x1 - x0) * pad_ratio)))
    pad_y = max(1, int(round((y1 - y0) * pad_ratio)))
    x0 -= pad_x
    y0 -= pad_y
    x1 += pad_x
    y1 += pad_y
    if x1 - x0 < min_side:
        cx = (x0 + x1) // 2
        x0 = cx - min_side // 2
        x1 = x0 + min_side
    if y1 - y0 < min_side:
        cy = (y0 + y1) // 2
        y0 = cy - min_side // 2
        y1 = y0 + min_side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    return (x0, y0, x1, y1)


def feather_radius_for(size) -> int:
    """羽化半徑隨原圖尺寸縮放。"""
    return max(2, min(48, int(round(min(size) * 0.01))))


def composite_back(
    original: Image.Image, generated_full: Image.Image, mask_bin: Image.Image
) -> Image.Image:
    """以羽化遮罩把生成結果貼回原圖。

    羽化只作用在遮罩邊緣的貼合帶；最後再以二值遮罩合成一次，
    保證非遮罩區像素與原圖逐位元相同、遮罩核心完全由生成結果覆蓋。
    """
    radius = feather_radius_for(original.size)
    alpha = mask_bin.filter(ImageFilter.GaussianBlur(radius))
    out = Image.composite(generated_full, original, alpha)
    out = Image.composite(out, original, mask_bin)
    return out


# ---------------------------------------------------------------------------
# 模型載入（P1-5 預載、P2-8 健壯性）
# ---------------------------------------------------------------------------
def friendly_error_message(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if (
        isinstance(exc, MemoryError)
        or type(exc).__name__ == "OutOfMemoryError"
        or "out of memory" in low
        or "can't allocate" in low
        or "cannot allocate" in low
        or "defaultcpuallocator" in low
    ):
        return (
            "記憶體不足：請降低「處理解析度」或「推論步數」後再試一次"
            "（一次生成張數也可改回 1 張）。"
        )
    if (
        isinstance(exc, OSError)
        or "connection" in low
        or "timed out" in low
        or "offline" in low
        or "gated" in low
        or "401" in low
        or "403" in low
        or "repository not found" in low
        or "no space" in low
    ):
        return (
            "模型下載／載入失敗：請檢查網路連線（需能連上 Hugging Face）"
            "與磁碟剩餘空間，確認後再試一次。"
        )
    return f"執行失敗：{msg[:300]}"


@lru_cache(maxsize=1)
def _load_pipe_cached(model_id: str):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["variant"] = "fp16"

    pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)

    # 換成 DPM++ 2M Karras：在高 Guidance 下收斂與畫質較穩（本工具的調校以
    # 「符合指示」優先）。假 pipe（測試）沒有 scheduler 時略過。
    if DPMSolverMultistepScheduler is not None:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        except Exception:
            pass

    # torch 2.x 已內建 SDPA，不再 enable_attention_slicing（舊作法反而拖慢）。
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe, device


def _ensure_pipe(model_id: str):
    """載入管線（非 Gradio 版）：失敗 raise ValueError(友善訊息)。"""
    model_id = model_id or DEFAULT_MODEL_ID
    _set_pipe_status("載入中", f"正在下載／載入 {model_id}…")
    try:
        pipe, device = _load_pipe_cached(model_id)
    except Exception as exc:
        friendly = friendly_error_message(exc)
        _set_pipe_status("載入失敗", friendly)
        raise ValueError(friendly)
    _set_pipe_status("已載入", f"{model_id}（裝置：{device}）")
    return pipe, device


def load_pipe(model_id: str):
    """載入管線；失敗時轉成友善的 gr.Error。"""
    try:
        return _ensure_pipe(model_id)
    except ValueError as exc:
        raise gr.Error(str(exc))


def preload_model(model_id, progress=gr.Progress()):
    """預先下載／載入模型，回傳狀態文字（失敗不 raise，顯示原因）。"""
    model_id = model_id or DEFAULT_MODEL_ID
    progress(0.05, desc="開始下載／載入模型…")
    try:
        pipe, device = load_pipe(model_id)
    except gr.Error as err:
        return f"模型狀態：載入失敗 — {getattr(err, 'message', None) or err}"
    progress(1.0, desc="模型載入完成")
    return f"模型狀態：已載入（裝置：{device}，模型：{model_id}）"


def preload_on_start(model_id, progress=gr.Progress()):
    yield "模型狀態：背景預載中…（第一次開啟需先下載模型，約數 GB，請稍候）"
    yield preload_model(model_id, progress)


# ---------------------------------------------------------------------------
# 生成取消（本機用模組層級旗標；雲端 worker 每個 job 各持一個 Event）
# ---------------------------------------------------------------------------
class GenerationCancelled(Exception):
    """使用者在生成中途按下「取消生成」。

    與一般錯誤分開處理：UI 層捕捉後以 gr.Info 友善提示「已取消生成」，
    不存半成品，畫面回到可再次生成的狀態。
    """


# 本機生成的取消旗標：edit_image 開始時 clear，「取消生成」按鈕 set。
_local_cancel_event = threading.Event()


def request_local_cancel():
    """要求取消目前進行中的本機生成（UI「取消生成」按鈕呼叫）。"""
    _local_cancel_event.set()


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("已取消生成")


# ---------------------------------------------------------------------------
# 生成核心（本機與雲端 worker 共用）
# ---------------------------------------------------------------------------
def expand_seeds(seed, count: int):
    """seed >= 0：第 i 張用 seed+i；seed = -1：先抽隨機 base 再連號。"""
    count = max(1, min(4, int(count)))
    try:
        base = int(seed) if seed is not None else -1
    except (TypeError, ValueError):
        base = -1
    if base < 0:
        base = random.SystemRandom().randint(0, 2**31 - 1)
    return [base + i for i in range(count)]


def _pipe_supports_step_callback(pipe) -> bool:
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    if "callback_on_step_end" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _run_inpaint(
    pipe,
    device,
    prompt,
    negative_prompt,
    image,
    mask_image,
    num_steps,
    guidance_scale,
    strength,
    seed_value,
    progress=None,
    progress_span=(0.0, 1.0),
    cancel_event=None,
):
    _raise_if_cancelled(cancel_event)
    generator = torch.Generator(
        device=device if device == "cuda" else "cpu"
    ).manual_seed(int(seed_value))
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        image=image,
        mask_image=mask_image,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=generator,
    )
    if _pipe_supports_step_callback(pipe):
        total = max(1, int(num_steps))
        lo, hi = progress_span

        def _on_step_end(pipe_, step_index, timestep, callback_kwargs):
            # 每一步先檢查取消旗標：被 set 就立刻中止整批生成。
            _raise_if_cancelled(cancel_event)
            if progress is not None:
                frac = (step_index + 1) / total
                progress(
                    lo + (hi - lo) * frac,
                    desc=f"生成中（步數 {step_index + 1}/{total}）",
                )
            return callback_kwargs

        kwargs["callback_on_step_end"] = _on_step_end
    return pipe(**kwargs).images[0]


def run_edit_core(src, mask_bin, params, progress_cb=None, cancel_event=None):
    """照片生成的純運算核心：本機 Gradio 層與雲端 worker 共用。

    src：PIL 圖（會轉 RGB）；mask_bin：二值遮罩（L，與 src 同尺寸）或 None。
    params：dict（edit_instruction / preserve_instruction / negative_prompt /
    model_id / num_steps / guidance_scale / strength / proc_side /
    num_images / seed / mask_source）。
    progress_cb(frac, desc) 可為 None。
    cancel_event：threading.Event；被 set 時在下一步邊界 raise
    GenerationCancelled（整批中止）。None 時用本機模組層級旗標。
    回傳 (results, used_mask, per_image_infos, aggregate)。
    錯誤一律 raise ValueError(友善中文訊息)；取消則 raise GenerationCancelled。
    """
    if cancel_event is None:
        cancel_event = _local_cancel_event

    def cb(frac, desc=None):
        if progress_cb is not None:
            progress_cb(frac, desc)

    _raise_if_cancelled(cancel_event)
    src = src.convert("RGB")
    orig_w, orig_h = src.size

    force_full = bool(params.get("full_mask"))
    if force_full:
        # 全圖遮罩：以全 1 遮罩處理、裁切範圍即全幅（忽略傳入的遮罩）。
        mask_bin = Image.new("L", src.size, color=255)
    elif mask_bin is not None:
        if mask_bin.size != src.size:
            mask_bin = mask_bin.resize(src.size, Image.Resampling.NEAREST)
        mask_bin = binarize_mask(mask_bin)
        if mask_coverage(mask_bin) >= FULL_MASK_COVERAGE:
            # 遮罩近乎全白 → 視為全圖模式
            mask_bin = None
    mode = "full" if mask_bin is None else "crop"
    if force_full:
        mask_source = "full_mask"
    elif mode == "full":
        mask_source = "full"
    else:
        mask_source = params.get("mask_source") or "mask"

    cb(0.05, "翻譯提示文字…")
    prompts = build_prompts(
        params.get("edit_instruction"),
        params.get("preserve_instruction"),
        params.get("negative_prompt"),
    )

    cb(0.08, "載入模型…")
    model_id = params.get("model_id") or DEFAULT_MODEL_ID
    pipe, device = _ensure_pipe(model_id)

    num_steps = params.get("num_steps", 50)
    guidance_scale = params.get("guidance_scale", 11)
    strength = params.get("strength", 1.0)
    proc_side = params.get("proc_side", DEFAULT_PROC_SIDE)
    count = max(1, min(4, int(params.get("num_images", 2))))
    seeds = expand_seeds(params.get("seed", -1), count)

    results = []
    per_image_infos = []
    used_mask = None
    for i, seed_value in enumerate(seeds):
        # 每張開始前檢查：第一張中途取消＝整批取消，後續張數不啟動。
        _raise_if_cancelled(cancel_event)
        lo = 0.1 + 0.85 * i / count
        hi = 0.1 + 0.85 * (i + 1) / count
        cb(lo, f"生成第 {i + 1}/{count} 張…")
        try:
            if mode == "full":
                pw, ph = normalize_size(orig_w, orig_h, max_side=int(proc_side))
                proc_img = src.resize((pw, ph), Image.Resampling.LANCZOS)
                proc_mask = Image.new("L", (pw, ph), color=255)
                gen = _run_inpaint(
                    pipe,
                    device,
                    prompts["prompt"],
                    prompts["negative_prompt"],
                    proc_img,
                    proc_mask,
                    num_steps,
                    guidance_scale,
                    strength,
                    seed_value,
                    progress=progress_cb,
                    progress_span=(lo, hi),
                    cancel_event=cancel_event,
                )
                composed = (
                    gen.resize(src.size, Image.Resampling.LANCZOS)
                    if gen.size != src.size
                    else gen
                )
                used_mask = Image.new("L", src.size, color=255)
                process_size = f"{pw}x{ph}"
            else:
                box = (
                    (0, 0, orig_w, orig_h)
                    if force_full
                    else compute_crop_box(mask_bin)
                )
                crop_img = src.crop(box)
                crop_mask = mask_bin.crop(box)
                tw, th = fit_long_side(crop_img.width, crop_img.height, int(proc_side))
                proc_img = crop_img.resize((tw, th), Image.Resampling.LANCZOS)
                proc_mask = crop_mask.resize((tw, th), Image.Resampling.NEAREST)
                gen = _run_inpaint(
                    pipe,
                    device,
                    prompts["prompt"],
                    prompts["negative_prompt"],
                    proc_img,
                    proc_mask,
                    num_steps,
                    guidance_scale,
                    strength,
                    seed_value,
                    progress=progress_cb,
                    progress_span=(lo, hi),
                    cancel_event=cancel_event,
                )
                gen_back = gen.resize(
                    (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS
                )
                gen_full = src.copy()
                gen_full.paste(gen_back, (box[0], box[1]))
                composed = composite_back(src, gen_full, mask_bin)
                used_mask = mask_bin
                process_size = f"{tw}x{th}"
        except GenerationCancelled:
            raise
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(friendly_error_message(exc))

        info_i = {
            "model": model_id,
            "device": device,
            "prompt": prompts["prompt"],
            "prompt_original": prompts["prompt_original"],
            "negative_prompt": prompts["negative_prompt"],
            "negative_prompt_original": prompts["negative_prompt_original"],
            "translation": prompts["translation"],
            "steps": int(num_steps),
            "guidance_scale": float(guidance_scale),
            "strength": float(strength),
            "proc_side": int(proc_side),
            "process_size": process_size,
            "size": f"{orig_w}x{orig_h}",
            "mode": mode,
            "mask_source": mask_source,
            "seed": int(seed_value),
            "index": i + 1,
        }
        results.append(composed)
        per_image_infos.append(info_i)

    aggregate = dict(per_image_infos[0])
    aggregate["seeds"] = seeds
    aggregate["count"] = count
    cb(1.0, "完成")
    return results, used_mask, per_image_infos, aggregate


def save_result(image: Image.Image, info: dict, stamp: str):
    """把結果存到 outputs/：edit_YYYYMMDD_HHMMSS_i.png ＋ 同名 .json。"""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    idx = int(info.get("index", 1))
    base = os.path.join(OUTPUTS_DIR, f"edit_{stamp}_{idx}")
    png_path = base + ".png"
    n = 1
    while os.path.exists(png_path):
        n += 1
        png_path = f"{base}_{n}.png"
    json_path = os.path.splitext(png_path)[0] + ".json"
    image.save(png_path)
    info["saved_to"] = png_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return png_path, json_path


def _resolve_mask(src, editor_value, uploaded_mask):
    """遮罩來源優先序：編輯器塗鴉 > 上傳遮罩 > 無（全圖模式）。

    回傳 (mask_bin 或 None, 來源字串 'editor'/'upload'/'full')。
    """
    mask_bin = mask_from_editor(editor_value, size=src.size)
    if mask_bin is not None:
        return mask_bin, "editor"
    up = pil_from_any(uploaded_mask)
    if up is not None:
        m = up.convert("L")
        if m.size != src.size:
            m = m.resize(src.size, Image.Resampling.LANCZOS)
        return binarize_mask(m), "upload"
    return None, "full"


def _params_from_ui(
    edit_instruction,
    preserve_instruction,
    negative_prompt,
    model_id,
    num_steps,
    guidance_scale,
    strength,
    proc_side,
    num_images,
    seed,
    full_mask=False,
    mode="mask",
):
    return {
        "edit_instruction": edit_instruction,
        "preserve_instruction": preserve_instruction,
        "negative_prompt": negative_prompt,
        "model_id": model_id,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "proc_side": proc_side,
        "num_images": num_images,
        "seed": seed,
        "full_mask": bool(full_mask),
        "mode": mode,
    }


def _save_and_pack(results, per_image_infos, aggregate, src, stamp):
    """把核心回傳的結果存進本機 outputs/，並組出 UI 用的 gallery/main_info。"""
    saved = []
    for img, info_i in zip(results, per_image_infos):
        save_result(img, info_i, stamp)
        saved.append(info_i["saved_to"])
    gallery = [
        (img, f"第 {i + 1} 張 · seed {per_image_infos[i]['seed']}")
        for i, img in enumerate(results)
    ]
    main_info = dict(aggregate)
    main_info["saved_files"] = saved
    return gallery, (src, results[0]), main_info


def edit_image(
    source_image,
    editor_value,
    uploaded_mask,
    edit_instruction,
    preserve_instruction,
    negative_prompt,
    model_id,
    num_steps,
    guidance_scale,
    strength,
    proc_side,
    num_images,
    seed,
    full_mask=False,
    progress=gr.Progress(),
):
    """本機 in-process 生成（Gradio 包裝層；行為與舊版相同）。"""
    _local_cancel_event.clear()
    src = pil_from_any(source_image)
    if src is None:
        raise gr.Error("請先上傳原始照片。")
    src = src.convert("RGB")

    progress(0.02, desc="準備圖片與遮罩…")
    mask_bin, mask_source = _resolve_mask(src, editor_value, uploaded_mask)
    params = _params_from_ui(
        edit_instruction,
        preserve_instruction,
        negative_prompt,
        model_id,
        num_steps,
        guidance_scale,
        strength,
        proc_side,
        num_images,
        seed,
        full_mask=full_mask,
    )
    params["mask_source"] = mask_source
    try:
        results, used_mask, per_infos, aggregate = run_edit_core(
            src, mask_bin, params, progress_cb=progress
        )
    except ValueError as exc:
        raise gr.Error(str(exc))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gallery, compare, main_info = _save_and_pack(
        results, per_infos, aggregate, src, stamp
    )
    return gallery, compare, used_mask, main_info


# ---------------------------------------------------------------------------
# 雲端 worker 串接（CloudClient / 隧道 / 連線測試）
# ---------------------------------------------------------------------------
def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _encode_multipart(fields, files):
    """組 multipart/form-data。fields: {name: str}；files: {name: (檔名, bytes, 類型)}。"""
    boundary = "----photoeditor" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode("utf-8")
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        body += str(value).encode("utf-8") + b"\r\n"
    for name, (filename, data, ctype) in files.items():
        body += f"--{boundary}\r\n".encode("utf-8")
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        ).encode("utf-8")
        body += f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


class CloudClient:
    """與雲端 worker.py 通訊：/health、POST /jobs、輪詢、/result 下載。"""

    def __init__(
        self,
        base_url,
        token="",
        health_timeout=5,
        request_timeout=60,
        poll_interval=0.8,
        job_timeout=3600,
    ):
        self.base_url = (base_url or DEFAULT_WORKER_URL).strip().rstrip("/")
        self.token = token or ""
        self.health_timeout = health_timeout
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self.current_job_id = None

    def _request(self, method, path, data=None, headers=None, timeout=None):
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if self.token:
            req.add_header("X-Worker-Token", self.token)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(
                req, timeout=timeout or self.request_timeout
            ) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            if exc.code == 401:
                raise ValueError(
                    "雲端 worker 認證失敗（401）：請確認「認證 Token」"
                    "與 worker 啟動時的 WORKER_TOKEN 是同一組。"
                )
            raise ValueError(f"雲端 worker 回應錯誤（HTTP {exc.code}）：{detail}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            raise ValueError(
                "連不上雲端 worker：請確認 worker 已在雲端主機啟動、網址正確；"
                "若透過 GPUtw 隧道連線，請先按「建立隧道」。"
            )

    def health(self):
        _status, body = self._request(
            "GET", "/health", timeout=self.health_timeout
        )
        return json.loads(body.decode("utf-8"))

    def submit_job(self, src, mask_bin, params):
        files = {"source": ("source.png", _pil_to_png_bytes(src), "image/png")}
        if mask_bin is not None:
            files["mask"] = ("mask.png", _pil_to_png_bytes(mask_bin), "image/png")
        payload, ctype = _encode_multipart(
            {"params": json.dumps(params, ensure_ascii=False)}, files
        )
        _status, body = self._request(
            "POST",
            "/jobs",
            data=payload,
            headers={"Content-Type": ctype},
            timeout=60,
        )
        return json.loads(body.decode("utf-8"))["job_id"]

    def get_job(self, job_id):
        _status, body = self._request("GET", f"/jobs/{job_id}", timeout=30)
        return json.loads(body.decode("utf-8"))

    def cancel_job(self, job_id):
        """通知 worker 取消指定工作（POST /jobs/{id}/cancel）。"""
        _status, body = self._request(
            "POST", f"/jobs/{job_id}/cancel", data=b"", timeout=30
        )
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return {}

    def download_result(self, job_id):
        _status, body = self._request(
            "GET", f"/jobs/{job_id}/result", timeout=300
        )
        zf = zipfile.ZipFile(io.BytesIO(body))
        meta = json.loads(zf.read("info.json").decode("utf-8"))
        results = []
        for name in sorted(n for n in zf.namelist() if n.endswith(".png")):
            results.append(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
        return results, meta.get("aggregate", {}), meta.get("per_image", [])

    def run_job(self, src, mask_bin, params, progress_cb=None, on_job_id=None):
        """送出工作 → 輪詢進度 → 下載結果。回傳 (results, aggregate, per_image_infos)。

        on_job_id(job_id) 在工作送出後立刻回報（供 UI 的取消按鈕找到目標）；
        輪詢到 status=cancelled 時 raise GenerationCancelled。
        """
        job_id = self.submit_job(src, mask_bin, params)
        self.current_job_id = job_id
        if on_job_id is not None:
            on_job_id(job_id)
        started = time.time()
        while True:
            st = self.get_job(job_id)
            if progress_cb is not None:
                progress_cb(
                    float(st.get("progress") or 0.0),
                    st.get("desc") or "雲端生成中…",
                )
            if st.get("status") == "done":
                break
            if st.get("status") == "cancelled":
                raise GenerationCancelled("已取消生成")
            if st.get("status") == "error":
                raise ValueError(
                    "雲端 worker 生成失敗：" + (st.get("error") or "未知錯誤")
                )
            if time.time() - started > self.job_timeout:
                raise ValueError("雲端生成逾時：請稍後到雲端主機確認 worker 狀態。")
            time.sleep(self.poll_interval)
        return self.download_result(job_id)


def test_worker_connection(worker_url, worker_token):
    """測試連線：回傳 (狀態文字, health dict 或 None)。"""
    client = CloudClient(worker_url, worker_token)
    try:
        health = client.health()
    except ValueError as exc:
        return f"連線失敗：{exc}", None
    caps = health.get("capabilities") or {}
    photo = "可用" if caps.get("photo") else "不可用"
    video = "可用" if caps.get("video") else "尚未啟用"
    text = (
        f"連線成功：裝置 {health.get('device')}／模型 {health.get('model')}"
        f"／照片修圖 {photo}／影片換人 {video}"
    )
    return text, health


# 雲端生成進行中的 job_id（edit_image_routed 透過 run_job 的 on_job_id 寫入，
# 「取消生成」按鈕讀它來通知 worker 取消；生成結束後清除）。
_current_cloud_job = {"job_id": None}


def on_cancel_generation(backend, worker_url, worker_token):
    """「取消生成」按鈕 handler（與 Gradio cancels 機制雙保險）。

    本機：set 模組取消旗標，核心在下一步邊界中止；
    雲端：若有進行中的工作，呼叫 worker 的取消端點。
    """
    request_local_cancel()
    job_id = _current_cloud_job.get("job_id")
    if backend == "cloud" and job_id:
        try:
            CloudClient(worker_url, worker_token).cancel_job(job_id)
        except ValueError:
            pass  # worker 端取消失敗不阻擋本機端（cancels 會停掉輪詢等待）
    gr.Info("已取消生成")


def edit_image_routed(
    backend,
    worker_url,
    worker_token,
    edit_mode,
    source_image,
    editor_value,
    uploaded_mask,
    full_mask,
    edit_instruction,
    preserve_instruction,
    negative_prompt,
    model_id,
    num_steps,
    guidance_scale,
    strength,
    proc_side,
    num_images,
    seed,
    progress=gr.Progress(),
):
    """依「運算位置」路由：本機走 in-process；雲端送 worker 執行。

    edit_mode："mask"（遮罩修圖）或 "instruction"（指令編輯，僅雲端）。
    回傳 (gallery, compare, info)，info 會標示 backend。
    """
    src = pil_from_any(source_image)
    if src is None:
        raise gr.Error("請先上傳原始照片。")
    src = src.convert("RGB")

    if edit_mode == "instruction" and backend != "cloud":
        raise gr.Error("本機不支援指令編輯，請用遮罩修圖或切換到雲端 GPU。")

    if backend != "cloud":
        try:
            gallery, compare, _used_mask, info = edit_image(
                source_image,
                editor_value,
                uploaded_mask,
                edit_instruction,
                preserve_instruction,
                negative_prompt,
                model_id,
                num_steps,
                guidance_scale,
                strength,
                proc_side,
                num_images,
                seed,
                full_mask=full_mask,
                progress=progress,
            )
        except GenerationCancelled:
            gr.Info("已取消生成")
            return (
                gr.update(),
                gr.update(),
                {"cancelled": True, "message": "已取消生成", "backend": "local"},
            )
        info = dict(info)
        info["backend"] = "local"
        return gallery, compare, info

    client = CloudClient(worker_url, worker_token)
    if edit_mode == "instruction":
        # 指令編輯：先向 worker 確認 capability，未啟用就擋下並說明。
        try:
            health = client.health()
        except ValueError as exc:
            raise gr.Error(str(exc))
        caps = health.get("capabilities") or {}
        if not caps.get("instruction"):
            raise gr.Error(
                "雲端 worker 尚未啟用指令編輯（capabilities.instruction=false）："
                "需要雲端 GPU 且 worker 需載入指令編輯模型，串接後自動啟用。"
            )
        mask_bin, mask_source = None, "instruction"
    elif full_mask:
        # 全圖遮罩：遮罩由 worker 端核心依旗標產生，不必傳遮罩圖。
        mask_bin, mask_source = None, "full_mask"
    else:
        progress(0.02, desc="準備圖片與遮罩…")
        mask_bin, mask_source = _resolve_mask(src, editor_value, uploaded_mask)
    params = _params_from_ui(
        edit_instruction,
        preserve_instruction,
        negative_prompt,
        model_id,
        num_steps,
        guidance_scale,
        strength,
        proc_side,
        num_images,
        seed,
        full_mask=full_mask,
        mode=edit_mode,
    )
    params["mask_source"] = mask_source
    try:
        results, aggregate, per_infos = client.run_job(
            src,
            mask_bin,
            params,
            progress_cb=progress,
            on_job_id=lambda jid: _current_cloud_job.update(job_id=jid),
        )
    except GenerationCancelled:
        gr.Info("已取消生成")
        return (
            gr.update(),
            gr.update(),
            {"cancelled": True, "message": "已取消生成", "backend": "cloud"},
        )
    except ValueError as exc:
        raise gr.Error(str(exc))
    finally:
        _current_cloud_job["job_id"] = None
    if not results:
        raise gr.Error("雲端 worker 沒有回傳任何結果。")

    def _stamp_source(info):
        if edit_mode == "instruction":
            return "instruction"
        if info.get("mask_source") == "full_mask":
            return "full_mask"
        return "full" if info.get("mode") == "full" else mask_source

    # 比照本機：成品存入本機 outputs/（PNG＋sidecar），sidecar 標示 backend=cloud
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    stamped_infos = []
    for i, img in enumerate(results):
        info_i = dict(per_infos[i]) if i < len(per_infos) else {"index": i + 1}
        info_i["index"] = i + 1
        info_i["backend"] = "cloud"
        info_i["mask_source"] = _stamp_source(info_i)
        save_result(img, info_i, stamp)
        saved.append(info_i["saved_to"])
        stamped_infos.append(info_i)
    gallery = [
        (img, f"第 {i + 1} 張 · seed {stamped_infos[i].get('seed')}")
        for i, img in enumerate(results)
    ]
    main_info = dict(aggregate)
    main_info["backend"] = "cloud"
    main_info["mask_source"] = _stamp_source(main_info)
    if "seeds" not in main_info:
        main_info["seeds"] = [info.get("seed") for info in stamped_infos]
    if "count" not in main_info:
        main_info["count"] = len(results)
    main_info["saved_files"] = saved
    progress(1.0, desc="完成")
    return gallery, (src, results[0]), main_info


# ---------------------------------------------------------------------------
# SSH 隧道（GPUtw：ssh -N -L 8001:127.0.0.1:8001 -p 2222 pod-<ID>@ssh.gputw.ai）
# ---------------------------------------------------------------------------
_tunnel = {"proc": None}


def build_tunnel_command(instance_id, local_port=8001, remote_port=8001):
    return [
        "ssh",
        "-N",
        "-L",
        f"{int(local_port)}:127.0.0.1:{int(remote_port)}",
        "-p",
        "2222",
        f"pod-{instance_id}@ssh.gputw.ai",
    ]


def _port_from_url(url):
    try:
        return urllib.parse.urlparse(url).port or 8001
    except Exception:
        return 8001


def start_tunnel(instance_id, worker_url=""):
    instance_id = (instance_id or "").strip()
    if not instance_id:
        return "隧道未建立：請先在「執行個體 ID」欄位填入 GPUtw 的執行個體 ID。"
    proc = _tunnel.get("proc")
    if proc is not None and proc.poll() is None:
        return "隧道已在執行中。"
    port = _port_from_url(worker_url) if worker_url else 8001
    cmd = build_tunnel_command(instance_id, local_port=port, remote_port=port)
    try:
        _tunnel["proc"] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as exc:
        _tunnel["proc"] = None
        return (
            f"隧道建立失敗：無法啟動 ssh（{exc}）。"
            "Windows 10/11 內建 OpenSSH 用戶端，請確認已啟用。"
        )
    return f"隧道建立中（{' '.join(cmd)}），請稍候幾秒再按「測試連線」。"


def stop_tunnel():
    proc = _tunnel.get("proc")
    if proc is None or proc.poll() is not None:
        _tunnel["proc"] = None
        return "隧道未建立（或已結束）。"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    _tunnel["proc"] = None
    return "隧道已關閉。"


def tunnel_status_text():
    proc = _tunnel.get("proc")
    if proc is not None and proc.poll() is None:
        return "隧道執行中（ssh 背景程序運行中）。"
    return "隧道未建立。"


# ---------------------------------------------------------------------------
# 影片換人（UI 與驗證先行；生成待雲端 worker 影片功能串接）
# ---------------------------------------------------------------------------
def _video_path(value):
    """從 gr.Video 的值取出檔案路徑（相容 str / dict / tuple 等形狀）。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _video_path(item)
            if path:
                return path
        return None
    if isinstance(value, dict):
        for key in ("video", "path", "name"):
            if key in value:
                path = _video_path(value[key])
                if path:
                    return path
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


def get_video_duration(path) -> float:
    """以 imageio 讀取影片長度（秒）；讀不到 raise ValueError。"""
    import imageio

    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        duration = meta.get("duration")
        if duration:
            return float(duration)
        fps = meta.get("fps") or 0
        if fps:
            return reader.count_frames() / float(fps)
    finally:
        reader.close()
    raise ValueError("無法讀取影片長度")


def extract_video_frame(path, fraction=0.5) -> Image.Image:
    """抽取影片在 fraction 位置的一幀（PIL RGB）。"""
    import imageio

    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps") or 24
        duration = meta.get("duration") or 0
        idx = 0
        if duration:
            frac = max(0.0, min(0.99, float(fraction)))
            idx = int(frac * duration * fps)
        frame = reader.get_data(idx)
    finally:
        reader.close()
    return Image.fromarray(frame).convert("RGB")


def normalize_click(x, y, width, height):
    """把像素座標正規化為 0–1（超出範圍夾住）。"""
    if width <= 0 or height <= 0:
        return (0.0, 0.0)
    return (
        min(max(float(x) / float(width), 0.0), 1.0),
        min(max(float(y) / float(height), 0.0), 1.0),
    )


def draw_marker(frame: Image.Image, nx: float, ny: float) -> Image.Image:
    """在幀圖的正規化座標 (nx, ny) 畫上明顯的紅色十字標記，回傳新圖。"""
    img = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = nx * w, ny * h
    r = max(8, int(min(w, h) * 0.03))
    lw = max(2, r // 3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 60, 60), width=lw)
    draw.line([cx - r * 1.8, cy, cx + r * 1.8, cy], fill=(255, 60, 60), width=lw)
    draw.line([cx, cy - r * 1.8, cx, cy + r * 1.8], fill=(255, 60, 60), width=lw)
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 60, 60))
    return img


def video_ready(backend, health):
    """影片生成可否啟用：回傳 (bool, 不可用的原因)。"""
    if backend != "cloud":
        return False, "影片換人需要雲端 GPU：請先把「運算位置」切換到雲端 GPU。"
    if not health or not health.get("ok"):
        return False, "尚未確認雲端連線：請先在「雲端設定」按「測試連線」。"
    caps = health.get("capabilities") or {}
    if not caps.get("video"):
        return False, "雲端 worker 的影片功能尚未啟用（待 Wan Animate＋SAM2 串接）。"
    return True, ""


def on_video_upload(value):
    """上傳影片後：驗證時長（>65 秒報友善錯誤）並抽取中間幀供點選。"""
    path = _video_path(value)
    if not path:
        return None, None, None, "尚未選擇", "請先上傳影片。"
    try:
        duration = get_video_duration(path)
    except Exception as exc:
        raise gr.Error(f"讀取影片失敗：{exc}")
    if duration > VIDEO_MAX_SECONDS:
        raise gr.Error(
            f"影片長度約 {duration:.0f} 秒，超過上限 {VIDEO_MAX_SECONDS} 秒："
            "請先把影片剪短到約 1 分鐘以內再上傳。"
        )
    try:
        frame = extract_video_frame(path, 0.5)
    except Exception as exc:
        raise gr.Error(f"抽取影片畫面失敗：{exc}")
    info = (
        f"影片長度約 {duration:.1f} 秒（上限 {VIDEO_MAX_SECONDS} 秒）。"
        "下方顯示的是影片中間畫面，請在畫面上點選要替換的人。"
    )
    return frame, frame, None, "尚未選擇", info


def on_frame_select(frame_state, evt: gr.SelectData):
    """在幀圖上點選：正規化座標、畫標記回顯、回報百分比文字。"""
    if frame_state is None:
        return None, None, "尚未選擇（請先上傳影片）"
    w, h = frame_state.size
    x, y = evt.index[0], evt.index[1]
    nx, ny = normalize_click(x, y, w, h)
    marked = draw_marker(frame_state, nx, ny)
    return marked, (nx, ny), f"已選擇位置 ({nx * 100:.0f}%, {ny * 100:.0f}%)"


def on_video_generate(
    backend, health, video_value, photo, selection, resolution, keep_audio
):
    ok, reason = video_ready(backend, health)
    if not ok:
        gr.Info(f"影片換人需要雲端 GPU，worker 影片功能串接後啟用。{reason}")
        return
    gr.Info("影片換人生成將於雲端串接完成後啟用。")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _editor_background(img):
    """把上傳的原圖帶入遮罩編輯器當底圖（回傳單張圖即設為 background）。"""
    return img


def _persist_settings(backend, worker_url, worker_token, instance_id):
    save_settings(
        {
            "backend": backend,
            "worker_url": worker_url,
            "worker_token": worker_token,
            "instance_id": instance_id,
        }
    )


def _on_backend_change(backend, worker_url, worker_token, instance_id):
    _persist_settings(backend, worker_url, worker_token, instance_id)
    return backend_status_text(backend)


def _on_cloud_fields_change(backend, worker_url, worker_token, instance_id):
    _persist_settings(backend, worker_url, worker_token, instance_id)
    return None


def example_texts():
    return [
        [
            "將人物上衣更換為黑色合身西裝外套，保留臉部、髮型、姿勢與背景，整體維持真實攝影風格。",
            "保留人物的臉部特徵、髮型、手部與原本姿勢，不要改變背景。",
            "low quality, blurry, deformed, extra fingers, extra limbs, unrealistic cloth",
        ],
        [
            "把人物改成白色襯衫加深藍色牛仔外套，風格自然、乾淨、日常穿搭。",
            "保留人物身分與背景，只修改上半身服裝。",
            "blurry, bad anatomy, weird hands, duplicate body parts",
        ],
        [
            "將背景改成現代咖啡廳，人物穿米色風衣與白色高領毛衣，整體偏時尚雜誌攝影。",
            "保留人物五官與身形比例自然。",
            "cartoon, anime, low detail, distorted face",
        ],
    ]


APP_CSS = """
.hint-box {padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(59,130,246,0.45); background: rgba(59,130,246,0.12); color: inherit;}
.warn-box {padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(245,158,11,0.5); background: rgba(245,158,11,0.13); color: inherit;}
.guarantee-box {padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(34,197,94,0.45); background: rgba(34,197,94,0.12); color: inherit;}
.small-note {font-size: 0.92rem; color: inherit; opacity: 0.78;}
"""


# ---------------------------------------------------------------------------
# 指令編輯模式
# ---------------------------------------------------------------------------
EDIT_INSTRUCTION_LABEL_MASK = (
    "③ 輸入修圖指示（中文即可，會自動翻譯成英文送給模型；中英混寫也可以）"
)
EDIT_INSTRUCTION_LABEL_INSTRUCTION = (
    "③ 輸入編輯指令（免遮罩：直接描述想把整張照片改成什麼樣子）"
)


def instruction_ready(backend, health):
    """指令編輯的啟用條件：雲端＋連線測試通過＋worker capabilities.instruction。

    回傳 (bool, 未就緒時的中文原因)。
    """
    if backend != "cloud":
        return False, "目前運算位置是本機：請先把「運算位置」切換到雲端 GPU。"
    if not health or not health.get("ok"):
        return False, "尚未確認雲端連線：請先在上方「雲端設定」按「測試連線」。"
    caps = health.get("capabilities") or {}
    if not caps.get("instruction"):
        return False, "目前連線的 worker 尚未啟用指令編輯（capabilities.instruction=false）。"
    return True, ""


def _on_edit_mode_change(edit_mode, backend, health):
    """切換編輯模式：指令編輯時隱藏遮罩區、改 prompt 標籤；未就緒時跳 Info 說明。"""
    if edit_mode == "instruction":
        ok, reason = instruction_ready(backend, health)
        if not ok:
            gr.Info(
                "指令編輯（免遮罩）需要雲端 GPU，且 worker 需載入指令編輯模型，"
                f"串接完成後會自動啟用。{reason}"
            )
        return (
            gr.update(visible=False),
            gr.update(label=EDIT_INSTRUCTION_LABEL_INSTRUCTION),
        )
    return (
        gr.update(visible=True),
        gr.update(label=EDIT_INSTRUCTION_LABEL_MASK),
    )


def build_ui():
    settings = load_settings()
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "上傳一張圖、畫遮罩、輸入 prompt，就能產出想要的效果。"
        )
        gr.Markdown(
            "<div class='hint-box'>"
            "遮罩塗到哪裡改哪裡：沒塗到的區域<b>逐位元貼回原圖</b>，絕不變形、不糊化；"
            "不塗遮罩＝全圖重繪。中文指示會自動翻譯成英文送給模型。"
            "</div>"
        )

        with gr.Row():
            backend_radio = gr.Radio(
                choices=BACKEND_CHOICES,
                value=settings["backend"],
                label="運算位置",
                scale=1,
            )
            backend_status = gr.Textbox(
                label="狀態",
                value=backend_status_text(settings["backend"]),
                interactive=False,
                scale=2,
            )

        with gr.Accordion("雲端設定（GPUtw worker）", open=False):
            worker_url = gr.Textbox(
                label="Worker 網址",
                value=settings["worker_url"],
                placeholder=DEFAULT_WORKER_URL,
            )
            worker_token = gr.Textbox(
                label="認證 Token（與 worker 啟動時的 WORKER_TOKEN 相同）",
                value=settings["worker_token"],
                type="password",
            )
            instance_id = gr.Textbox(
                label="執行個體 ID（GPUtw，選填；用於一鍵建立 SSH 隧道）",
                value=settings["instance_id"],
            )
            with gr.Row():
                test_conn_btn = gr.Button("測試連線")
                tunnel_start_btn = gr.Button("建立隧道")
                tunnel_stop_btn = gr.Button("關閉隧道")
            cloud_status = gr.Textbox(
                label="雲端連線狀態", value="尚未測試連線", interactive=False
            )
            tunnel_status = gr.Textbox(
                label="隧道狀態", value="隧道未建立", interactive=False
            )

        health_state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("照片修圖"):
                edit_mode = gr.Radio(
                    choices=[
                        ("遮罩修圖", "mask"),
                        ("指令編輯（免遮罩・雲端）", "instruction"),
                    ],
                    value="mask",
                    label="編輯模式",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        source_image = gr.Image(label="① 上傳照片", type="pil")
                        with gr.Column() as mask_area:
                            mask_editor = gr.ImageEditor(
                                label="② 塗遮罩（在照片上用白色筆刷塗出要修改的區域；不塗＝全圖模式）",
                                type="pil",
                                image_mode="RGBA",
                                brush=gr.Brush(
                                    colors=["#FFFFFF"],
                                    default_color="#FFFFFF",
                                    color_mode="fixed",
                                ),
                                eraser=gr.Eraser(),
                                sources=("upload", "clipboard"),
                            )
                            full_mask_cb = gr.Checkbox(
                                label="全圖遮罩（整張圖重繪・自由度最高）",
                                value=False,
                            )
                            full_mask_warn = gr.Markdown(
                                "<div class='warn-box'>"
                                "⚠️ <b>全圖遮罩已開啟</b>：整張圖都會重繪，自由度最高"
                                "（換裝、換背景、改姿勢都靠它）；但未編輯區<b>不再逐位元"
                                "保證原樣</b>，臉部與手部可能變形。開啟時會忽略手塗遮罩。"
                                "</div>",
                                visible=False,
                            )
                    with gr.Column(scale=1):
                        edit_instruction = gr.Textbox(
                            label="③ 輸入修圖指示（中文即可，會自動翻譯成英文送給模型；中英混寫也可以）",
                            lines=8,
                            placeholder="例如：將人物上衣換成黑色合身西裝外套，保留臉部、髮型、姿勢與背景，整體維持真實攝影風格。",
                        )
                        with gr.Row():
                            run_btn = gr.Button("開始修圖", variant="primary")
                            cancel_btn = gr.Button("取消生成")
                        gr.Markdown(
                            "<div class='small-note'>"
                            "本機參數已依你的電腦調校為最重視符合指示的設定，生成較慢屬正常。"
                            "第一次使用會先下載模型（數 GB），進度看「進階設定」裡的模型狀態；"
                            "可以按了讓它掛著跑。"
                            "</div>"
                        )

                with gr.Accordion("進階設定", open=False):
                    preserve_instruction = gr.Textbox(
                        label="保留內容（建議填寫，會接在指示後一起送出）",
                        lines=3,
                        placeholder="例如：保留人物的臉部特徵、髮型、手部與背景，不要改變構圖。",
                    )
                    negative_prompt = gr.Textbox(
                        label="避免內容（Negative Prompt）",
                        lines=3,
                        placeholder="例如：blurry, low quality, distorted face, bad hands, extra fingers",
                        value="blurry, low quality, bad anatomy, bad hands, deformed hands, extra fingers, extra limbs, censored, mosaic",
                    )
                    model_id = gr.Dropdown(
                        choices=list(MODEL_HINT.keys()),
                        value=DEFAULT_MODEL_ID,
                        label="模型",
                        info="目前僅提供已驗證可公開下載的 SD 1.5 Inpainting 模型。",
                    )
                    with gr.Row():
                        model_status = gr.Textbox(
                            label="模型狀態",
                            value="模型狀態：尚未載入（開啟頁面後會自動背景預載）",
                            interactive=False,
                            scale=3,
                        )
                        preload_btn = gr.Button("預先載入模型", scale=1)
                    with gr.Row():
                        num_steps = gr.Slider(10, 50, value=50, step=1, label="推論步數（越高越慢）")
                        guidance_scale = gr.Slider(1.0, 15.0, value=11, step=0.5, label="Guidance Scale")
                    with gr.Row():
                        strength = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="修改強度")
                        proc_side = gr.Slider(
                            384,
                            1024,
                            value=DEFAULT_PROC_SIDE,
                            step=32,
                            label="處理解析度（生成時的工作邊長；越高越精細、越久。遮罩模式下只處理塗抹區域附近，未編輯區維持原圖畫質）",
                        )
                    with gr.Row():
                        num_images = gr.Slider(1, 4, value=2, step=1, label="一次生成張數")
                        seed = gr.Number(value=-1, label="Seed（輸入 -1 可隨機；多張時依序 +1）", precision=0)
                    uploaded_mask = gr.Image(
                        label="上傳遮罩圖（選填後備；白色 = 要修改，黑色 = 保留。編輯器有塗鴉時以編輯器為準）",
                        type="pil",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery(label="修圖結果", columns=2)
                        output_compare = gr.ImageSlider(
                            label="原圖 vs 修圖結果（拖動滑桿比較第一張）",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        output_info = gr.JSON(label="執行資訊")

                gr.Examples(
                    examples=example_texts(),
                    inputs=[edit_instruction, preserve_instruction, negative_prompt],
                    label="範例指令",
                )

            with gr.Tab("影片換人"):
                gr.Markdown(
                    "上傳約 1 分鐘的影片與一張人物照片，把影片中的人物換成照片裡的人"
                    "（髮型、身形、樣貌整體替換；影片中有兩個人時，可點選要換的那一位）。\n\n"
                    "**目前狀態**：介面與選人流程已就緒；實際生成需要雲端 GPU，"
                    "且 worker 的影片功能（Wan Animate＋SAM2 追蹤）串接後才會啟用，本機無法執行。"
                )
                video_input = gr.Video(
                    label="① 上傳影片（約 1 分鐘，最長 65 秒）", sources=["upload"]
                )
                video_info_text = gr.Textbox(
                    label="影片資訊", value="尚未上傳影片", interactive=False
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_photo = gr.Image(label="② 上傳人物照片", type="pil")
                    with gr.Column(scale=1):
                        frame_image = gr.Image(
                            label="③ 點選要替換的人（在畫面上點一下目標人物）",
                            type="pil",
                            interactive=False,
                        )
                        selection_text = gr.Textbox(
                            label="已選擇位置", value="尚未選擇", interactive=False
                        )
                frame_state = gr.State(None)
                selection_state = gr.State(None)
                with gr.Row():
                    video_res = gr.Radio(
                        choices=["480p", "720p"], value="720p", label="解析度"
                    )
                    video_keep_audio = gr.Checkbox(value=True, label="保留原音")
                video_run_btn = gr.Button("開始換人（需雲端 GPU）", variant="primary")

        # ---------------- 事件 wiring ----------------
        source_image.upload(
            fn=_editor_background, inputs=source_image, outputs=mask_editor
        )

        preload_btn.click(fn=preload_model, inputs=[model_id], outputs=[model_status])

        demo.load(fn=preload_on_start, inputs=[model_id], outputs=[model_status])

        backend_radio.change(
            fn=_on_backend_change,
            inputs=[backend_radio, worker_url, worker_token, instance_id],
            outputs=[backend_status],
        )
        for _field in (worker_url, worker_token, instance_id):
            _field.change(
                fn=_on_cloud_fields_change,
                inputs=[backend_radio, worker_url, worker_token, instance_id],
            )

        test_conn_btn.click(
            fn=test_worker_connection,
            inputs=[worker_url, worker_token],
            outputs=[cloud_status, health_state],
        )
        tunnel_start_btn.click(
            fn=start_tunnel,
            inputs=[instance_id, worker_url],
            outputs=[tunnel_status],
        )
        tunnel_stop_btn.click(fn=stop_tunnel, outputs=[tunnel_status])

        edit_mode.change(
            fn=_on_edit_mode_change,
            inputs=[edit_mode, backend_radio, health_state],
            outputs=[mask_area, edit_instruction],
        )
        full_mask_cb.change(
            fn=lambda on: gr.update(visible=bool(on)),
            inputs=[full_mask_cb],
            outputs=[full_mask_warn],
        )
        generate_event = run_btn.click(
            fn=edit_image_routed,
            inputs=[
                backend_radio,
                worker_url,
                worker_token,
                edit_mode,
                source_image,
                mask_editor,
                uploaded_mask,
                full_mask_cb,
                edit_instruction,
                preserve_instruction,
                negative_prompt,
                model_id,
                num_steps,
                guidance_scale,
                strength,
                proc_side,
                num_images,
                seed,
            ],
            outputs=[output_gallery, output_compare, output_info],
            concurrency_limit=1,
        )
        # 取消生成（雙保險）：handler 先 set 取消旗標／通知 worker 取消，
        # 同時以 Gradio cancels 機制取消正在跑的生成事件，讓 UI 立刻解鎖。
        cancel_btn.click(
            fn=on_cancel_generation,
            inputs=[backend_radio, worker_url, worker_token],
            cancels=[generate_event],
        )

        video_input.upload(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[
                frame_image,
                frame_state,
                selection_state,
                selection_text,
                video_info_text,
            ],
        )
        frame_image.select(
            fn=on_frame_select,
            inputs=[frame_state],
            outputs=[frame_image, selection_state, selection_text],
        )
        video_run_btn.click(
            fn=on_video_generate,
            inputs=[
                backend_radio,
                health_state,
                video_input,
                video_photo,
                selection_state,
                video_res,
                video_keep_audio,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=APP_CSS,
    )
