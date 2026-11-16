"""雲端 GPU worker：把照片修圖核心（app.run_edit_core）包成 HTTP 服務。

在雲端主機上啟動：
    pip install -r requirements.txt
    pip install -r requirements-worker.txt
    WORKER_TOKEN=你的密鑰 python worker.py          # 監聽 127.0.0.1:8001

本機（Windows）再用 SSH 隧道連入：
    ssh -N -L 8001:127.0.0.1:8001 -p 2222 pod-<執行個體ID>@ssh.gputw.ai
主程式（app.py）的「雲端設定」填 Worker 網址 http://127.0.0.1:8001 與相同 Token。

環境變數：
    WORKER_TOKEN  認證密鑰（強烈建議設定；UI 的「認證 Token」要填同一組）
    WORKER_PORT   監聽埠，預設 8001
    WORKER_FAKE   =1 時用假 pipe（回傳標記色圖），不載入真模型（測試用）
    WORKER_FAKE_STEP_DELAY  假 pipe 每步延遲秒數（預設 0；取消功能測試用）

API：
    GET  /health           {ok, device, model, capabilities:{photo, video, instruction}}
    POST /jobs             multipart：source PNG、mask PNG（選填）、params JSON
                           （params.mode 可為 mask／instruction；instruction 未啟用回 400）
    GET  /jobs/{id}        {status, progress, desc, error}
    POST /jobs/{id}/cancel 取消工作：set 該 job 的取消旗標，job 在下一步邊界
                           中止並轉為 status=cancelled（error 欄位帶「已取消生成」）
    GET  /jobs/{id}/result zip：0.png…N.png ＋ info.json
    除 /health 外（health 有帶 token 才驗證），其餘端點需標頭 X-Worker-Token。
    capabilities.video 目前固定回 false：影片換人（Wan Animate＋SAM2）待串接。
    capabilities.instruction 目前固定回 false：指令編輯模型（免遮罩）待串接；
    WORKER_FAKE=1 的測試模式下回 true，讓主程式的模式路由可端到端測試。
"""

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import io
import json
import threading
import time
import uuid
import zipfile
from types import SimpleNamespace
from typing import Optional

from PIL import Image
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

import app as editor

TOKEN = os.environ.get("WORKER_TOKEN", "")
FAKE = os.environ.get("WORKER_FAKE") == "1"
# 指令編輯（免遮罩）本輪僅在測試模式視為可用；正式串接指令編輯模型後改為 True。
INSTRUCTION_OK = FAKE
FAKE_COLOR = (222, 40, 160)

if FAKE:
    # 測試模式：假 pipe 回傳標記色圖；翻譯器停用（不連網下載）。
    FAKE_STEP_DELAY = float(os.environ.get("WORKER_FAKE_STEP_DELAY", "0") or 0)

    class _FakePipe:
        def to(self, device):
            return self

        def enable_vae_slicing(self):
            pass

        def __call__(self, **kwargs):
            img = kwargs["image"]
            # 模擬逐步生成：逐步呼叫 step callback，讓取消旗標有機會在
            # 步邊界被檢查（與真 pipe 的行為一致）；每步可延遲。
            cb = kwargs.get("callback_on_step_end")
            steps = int(kwargs.get("num_inference_steps", 1) or 1)
            for i in range(steps):
                if FAKE_STEP_DELAY:
                    time.sleep(FAKE_STEP_DELAY)
                if cb is not None:
                    cb(self, i, i, {})
            return SimpleNamespace(images=[Image.new("RGB", img.size, FAKE_COLOR)])

    class _FakeAutoPipeline:
        _pipe = _FakePipe()

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls._pipe

    editor.AutoPipelineForInpainting = _FakeAutoPipeline
    editor._load_pipe_cached.cache_clear()

    def _no_translator():
        raise RuntimeError("fake worker：翻譯器停用")

    editor._load_translator = _no_translator

api = FastAPI(title="AI 文字修圖 worker")
_jobs = {}
_jobs_lock = threading.Lock()


def _require_token(x_worker_token: Optional[str]):
    if TOKEN and (x_worker_token or "") != TOKEN:
        raise HTTPException(status_code=401, detail="認證失敗：X-Worker-Token 不正確")


@api.get("/health")
def health(x_worker_token: Optional[str] = Header(default=None)):
    if x_worker_token is not None:
        _require_token(x_worker_token)
    return {
        "ok": True,
        "device": "fake" if FAKE else editor.get_device(),
        "model": editor.DEFAULT_MODEL_ID,
        "capabilities": {
            "photo": True,
            "video": False,
            "instruction": INSTRUCTION_OK,
        },
    }


def _run_job(job_id, src_bytes, mask_bytes, params):
    def _update(**kw):
        with _jobs_lock:
            _jobs[job_id].update(kw)

    with _jobs_lock:
        cancel_event = (_jobs.get(job_id) or {}).get("cancel_event")

    _update(status="running", progress=0.0)
    try:
        src = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        mask_bin = None
        if mask_bytes:
            m = Image.open(io.BytesIO(mask_bytes)).convert("L")
            if m.size != src.size:
                m = m.resize(src.size, Image.Resampling.NEAREST)
            mask_bin = editor.binarize_mask(m)
        params = dict(params or {})
        params.setdefault("mask_source", "cloud")

        def _cb(frac, desc=None):
            _update(progress=float(frac), desc=desc or "")

        results, _used_mask, per_infos, aggregate = editor.run_edit_core(
            src, mask_bin, params, progress_cb=_cb, cancel_event=cancel_event
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(results):
                b = io.BytesIO()
                img.save(b, format="PNG")
                zf.writestr(f"{i}.png", b.getvalue())
            zf.writestr(
                "info.json",
                json.dumps(
                    {"aggregate": aggregate, "per_image": per_infos},
                    ensure_ascii=False,
                ),
            )
        _update(status="done", progress=1.0, result=buf.getvalue())
    except editor.GenerationCancelled:
        _update(status="cancelled", error="已取消生成")
    except ValueError as exc:
        _update(status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        _update(status="error", error=editor.friendly_error_message(exc))


@api.post("/jobs")
async def create_job(
    source: UploadFile = File(...),
    mask: UploadFile = File(None),
    params: str = Form("{}"),
    x_worker_token: Optional[str] = Header(default=None),
):
    _require_token(x_worker_token)
    src_bytes = await source.read()
    mask_bytes = await mask.read() if mask is not None else None
    try:
        params_dict = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params 不是合法 JSON")
    mode = params_dict.get("mode", "mask")
    if mode not in ("mask", "instruction"):
        raise HTTPException(
            status_code=400, detail="未知的編輯模式 mode（只接受 mask 或 instruction）"
        )
    if mode == "instruction" and not INSTRUCTION_OK:
        raise HTTPException(
            status_code=400,
            detail=(
                "此 worker 尚未啟用指令編輯（capabilities.instruction=false）："
                "需要雲端 GPU 並載入指令編輯模型，串接後開放。"
            ),
        )
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "desc": "",
            "error": None,
            "result": None,
            "cancel_event": threading.Event(),
        }
    threading.Thread(
        target=_run_job,
        args=(job_id, src_bytes, mask_bytes, params_dict),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@api.get("/jobs/{job_id}")
def get_job(
    job_id: str, x_worker_token: Optional[str] = Header(default=None)
):
    _require_token(x_worker_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到這個工作")
        return {
            "status": job["status"],
            "progress": job["progress"],
            "desc": job["desc"],
            "error": job["error"],
        }


@api.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str, x_worker_token: Optional[str] = Header(default=None)
):
    _require_token(x_worker_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到這個工作")
        ev = job.get("cancel_event")
        if ev is not None:
            ev.set()
        status = job["status"]
    return {"job_id": job_id, "status": status, "cancel_requested": True}


@api.get("/jobs/{job_id}/result")
def get_result(
    job_id: str, x_worker_token: Optional[str] = Header(default=None)
):
    _require_token(x_worker_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到這個工作")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail="工作尚未完成")
        data = job["result"]
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="result-{job_id}.zip"'
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        api, host="127.0.0.1", port=int(os.environ.get("WORKER_PORT", "8001"))
    )
