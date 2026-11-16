import inspect
import json
import os
import random
import re
import threading
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting

APP_TITLE = "AI 文字修圖工具（本機版）"
DEFAULT_MODEL_ID = os.environ.get(
    "MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"
)
DEFAULT_PROC_SIDE = int(os.environ.get("MAX_SIDE", "768"))
TRANSLATE_MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
QUALITY_SUFFIX = "high quality, detailed, photorealistic"
CJK_THRESHOLD = 0.3
FULL_MASK_COVERAGE = 0.98
EDITOR_MIN_MASK_PIXELS = 16
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MODEL_HINT = {
    DEFAULT_MODEL_ID: "通用修圖／換衣／背景替換（預設，SD 1.5 Inpainting）",
}

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

    # torch 2.x 已內建 SDPA，不再 enable_attention_slicing（舊作法反而拖慢）。
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe, device


def load_pipe(model_id: str):
    """載入管線；失敗時轉成友善的 gr.Error。"""
    _set_pipe_status("載入中", f"正在下載／載入 {model_id}…")
    try:
        pipe, device = _load_pipe_cached(model_id)
    except Exception as exc:
        friendly = friendly_error_message(exc)
        _set_pipe_status("載入失敗", friendly)
        raise gr.Error(friendly)
    _set_pipe_status("已載入", f"{model_id}（裝置：{device}）")
    return pipe, device


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
# 生成
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
):
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
    if progress is not None and _pipe_supports_step_callback(pipe):
        total = max(1, int(num_steps))
        lo, hi = progress_span

        def _on_step_end(pipe_, step_index, timestep, callback_kwargs):
            frac = (step_index + 1) / total
            progress(
                lo + (hi - lo) * frac,
                desc=f"生成中（步數 {step_index + 1}/{total}）",
            )
            return callback_kwargs

        kwargs["callback_on_step_end"] = _on_step_end
    return pipe(**kwargs).images[0]


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
    progress=gr.Progress(),
):
    src = pil_from_any(source_image)
    if src is None:
        raise gr.Error("請先上傳原始照片。")
    src = src.convert("RGB")
    orig_w, orig_h = src.size

    # --- 遮罩來源優先序：編輯器塗鴉 > 上傳遮罩 > 全圖模式 ---
    progress(0.02, desc="準備圖片與遮罩…")
    mask_bin = mask_from_editor(editor_value, size=src.size)
    mask_source = "editor" if mask_bin is not None else None
    if mask_bin is None:
        up = pil_from_any(uploaded_mask)
        if up is not None:
            m = up.convert("L")
            if m.size != src.size:
                m = m.resize(src.size, Image.Resampling.LANCZOS)
            mask_bin = binarize_mask(m)
            mask_source = "upload"
    if mask_bin is not None and mask_coverage(mask_bin) >= FULL_MASK_COVERAGE:
        # 遮罩近乎全白 → 視為全圖模式
        mask_bin = None
    mode = "full" if mask_bin is None else "crop"
    if mode == "full":
        mask_source = "full"

    # --- 翻譯與 prompt 組裝 ---
    progress(0.06, desc="翻譯提示文字…")
    prompts = build_prompts(edit_instruction, preserve_instruction, negative_prompt)

    # --- 載入模型 ---
    progress(0.08, desc="載入模型…")
    pipe, device = load_pipe(model_id)

    count = max(1, min(4, int(num_images)))
    seeds = expand_seeds(seed, count)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    per_image_infos = []
    used_mask = None
    for i, seed_value in enumerate(seeds):
        lo = 0.1 + 0.85 * i / count
        hi = 0.1 + 0.85 * (i + 1) / count
        progress(lo, desc=f"生成第 {i + 1}/{count} 張…")
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
                    progress=progress,
                    progress_span=(lo, hi),
                )
                composed = (
                    gen.resize(src.size, Image.Resampling.LANCZOS)
                    if gen.size != src.size
                    else gen
                )
                used_mask = Image.new("L", src.size, color=255)
                process_size = f"{pw}x{ph}"
            else:
                box = compute_crop_box(mask_bin)
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
                    progress=progress,
                    progress_span=(lo, hi),
                )
                gen_back = gen.resize(
                    (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS
                )
                gen_full = src.copy()
                gen_full.paste(gen_back, (box[0], box[1]))
                composed = composite_back(src, gen_full, mask_bin)
                used_mask = mask_bin
                process_size = f"{tw}x{th}"
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(friendly_error_message(exc))

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
        save_result(composed, info_i, stamp)
        results.append(composed)
        per_image_infos.append(info_i)

    progress(1.0, desc="完成")
    gallery = [
        (img, f"第 {i + 1} 張 · seed {per_image_infos[i]['seed']}")
        for i, img in enumerate(results)
    ]
    compare = (src, results[0])
    main_info = dict(per_image_infos[0])
    main_info["seeds"] = seeds
    main_info["count"] = count
    main_info["saved_files"] = [info["saved_to"] for info in per_image_infos]
    return gallery, compare, used_mask, main_info


def _editor_background(img):
    """把上傳的原圖帶入遮罩編輯器當底圖（回傳單張圖即設為 background）。"""
    return img


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
.small-note {font-size: 0.92rem; color: #666;}
.warn-box {padding: 10px 12px; border-radius: 10px; background: #fff8e1; color: #5d4037;}
.guarantee-box {padding: 10px 12px; border-radius: 10px; background: #e8f5e9; color: #1b5e20;}
"""


def build_ui():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "本工具使用 **開源 AI Inpainting 模型** 進行局部或整張照片修圖，適合換衣服、改背景、改風格、修人物細節。"
        )

        gr.Markdown(
            "<div class='guarantee-box'>"
            "<b>遮罩就是「指定哪裡能改、哪裡不准動」的工具。</b>"
            "有塗遮罩時，沒塗到的區域會由程式<b>逐位元貼回原圖</b>，絕不變形、不糊化"
            "——這是合成方式的架構保證，不是看模型運氣。"
            "生成的結果會以原圖的完整解析度輸出，只有遮罩範圍內的區域會被重畫。"
            "</div>"
        )

        gr.Markdown(
            "<div class='warn-box'>"
            "首次執行時會自動下載模型（可能需要數 GB 網路流量與儲存空間）。"
            "如果你目前是 <b>內顯/CPU</b> 執行，生成速度會比較慢：處理解析度 512 約需數分鐘起，"
            "768 / 1024 會更久，可以先按「開始修圖」後讓它掛著跑。"
            "不用遮罩也可以：遮罩留空就是<b>全圖模式</b>，任何提示都能改整張圖；"
            "但全圖重繪時，臉部、手部等細節的保真風險較高，建議先用遮罩模式比對效果。"
            "</div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                source_image = gr.Image(label="原始照片", type="pil")
                mask_editor = gr.ImageEditor(
                    label="遮罩編輯器（在照片上用白色筆刷塗出要修改的區域；不塗＝全圖模式）",
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
                uploaded_mask = gr.Image(
                    label="上傳遮罩圖（選填後備；白色 = 要修改，黑色 = 保留。編輯器有塗鴉時以編輯器為準）",
                    type="pil",
                )
                gr.Markdown(
                    "<div class='small-note'>"
                    "上傳原始照片後，它會自動成為遮罩編輯器的底圖，直接在照片上塗即可。"
                    "也可以沿用小畫家做的遮罩圖：白色塗要修改的位置，黑色保留不動。"
                    "例如只想換上衣，就在上衣位置畫白色，其餘畫黑色。"
                    "</div>"
                )

            with gr.Column(scale=1):
                edit_instruction = gr.Textbox(
                    label="修圖指示（可直接輸入完整中文，會自動翻譯成英文送給模型；中英混寫也可以）",
                    lines=6,
                    placeholder="例如：將人物上衣換成黑色合身西裝外套，保留臉部、髮型、姿勢與背景，整體維持真實攝影風格。",
                )
                preserve_instruction = gr.Textbox(
                    label="保留內容（建議填寫，會接在指示後一起送出）",
                    lines=3,
                    placeholder="例如：保留人物的臉部特徵、髮型、手部與背景，不要改變構圖。",
                )
                negative_prompt = gr.Textbox(
                    label="避免內容（Negative Prompt）",
                    lines=3,
                    placeholder="例如：blurry, low quality, distorted face, bad hands, extra fingers",
                    value="blurry, low quality, bad anatomy, bad hands, extra fingers, extra limbs",
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
                    num_steps = gr.Slider(10, 50, value=25, step=1, label="推論步數（越高越慢）")
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                with gr.Row():
                    strength = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="修改強度")
                    proc_side = gr.Slider(
                        512,
                        1024,
                        value=DEFAULT_PROC_SIDE,
                        step=64,
                        label="處理解析度（生成時的工作邊長；越高越精細、越久。遮罩模式下只處理塗抹區域附近，未編輯區維持原圖畫質）",
                    )
                with gr.Row():
                    num_images = gr.Slider(1, 4, value=1, step=1, label="一次生成張數")
                    seed = gr.Number(value=42, label="Seed（輸入 -1 可隨機；多張時依序 +1）", precision=0)

                run_btn = gr.Button("開始修圖", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                output_gallery = gr.Gallery(label="修圖結果", columns=2)
                output_compare = gr.ImageSlider(
                    label="原圖 vs 修圖結果（拖動滑桿比較第一張）", interactive=False
                )
            with gr.Column(scale=1):
                output_mask = gr.Image(label="實際使用的遮罩（全圖解析度）")
                output_info = gr.JSON(label="執行資訊")

        gr.Markdown("## 範例指令")
        gr.Examples(
            examples=example_texts(),
            inputs=[edit_instruction, preserve_instruction, negative_prompt],
        )

        source_image.upload(
            fn=_editor_background, inputs=source_image, outputs=mask_editor
        )

        preload_btn.click(fn=preload_model, inputs=[model_id], outputs=[model_status])

        demo.load(fn=preload_on_start, inputs=[model_id], outputs=[model_status])

        run_btn.click(
            fn=edit_image,
            inputs=[
                source_image,
                mask_editor,
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
            ],
            outputs=[output_gallery, output_compare, output_mask, output_info],
            concurrency_limit=1,
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
