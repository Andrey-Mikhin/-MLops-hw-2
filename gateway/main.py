from __future__ import annotations

import asyncio
import io
import os
import subprocess
import time
from pathlib import Path

import httpx
import jwt
import numpy as np
import tritonclient.http as httpclient
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

JWT_SECRET = os.environ["JWT_SECRET"]
STUDENT_NAME = os.environ.get("STUDENT_NAME", "")
STUDENT_EMAIL = os.environ["STUDENT_EMAIL"]
CV_MODEL = os.environ["CV_MODEL"]
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen2.5-1.5B-Instruct")

TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000")
TRITON_METRICS_URL = os.environ.get("TRITON_METRICS_URL", "http://triton:8002/metrics")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://vllm:8000/metrics")
NODE_METRICS_URL = os.environ.get("NODE_METRICS_URL", "http://node-exporter:9100/metrics")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RESIZE = 232
_CROP = 224


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((_RESIZE, _RESIZE), Image.BILINEAR)
    left = (img.width - _CROP) // 2
    top = (img.height - _CROP) // 2
    img = img.crop((left, top, left + _CROP, top + _CROP))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...].copy()


_CLASSES_PATH = Path(__file__).with_name("imagenet_classes.txt")
IMAGENET_CLASSES = [line.strip() for line in _CLASSES_PATH.read_text().splitlines() if line.strip()]
assert len(IMAGENET_CLASSES) == 1000, f"Ожидалось 1000 классов, получено {len(IMAGENET_CLASSES)}"

app = FastAPI(title="Real-time Inference Gateway")
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

_security = HTTPBearer(auto_error=False)


async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_security)) -> dict:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="missing token")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token")
    return payload


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{TRITON_URL}/v2/health/ready")
        ready = r.status_code == 200
    except httpx.HTTPError:
        ready = False
    if not ready:
        return Response(
            content=f'{{"backend": "triton", "ready": false, "model": "{CV_MODEL}"}}',
            status_code=503,
            media_type="application/json",
        )
    return {"backend": "triton", "ready": True, "model": CV_MODEL}


@app.get("/vllm/health")
async def vllm_health():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{VLLM_URL}/v1/models")
        ok = r.status_code == 200
        models = [m.get("id") for m in r.json().get("data", [])] if ok else []
    except (httpx.HTTPError, ValueError):
        ok, models = False, []
    if not ok:
        return Response(
            content='{"backend": "vllm", "ready": false, "models": []}',
            status_code=503,
            media_type="application/json",
        )
    return {"backend": "vllm", "ready": True, "models": models}


async def _proxy_metrics(url: str) -> Response:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"backend returned {r.status_code}")
        return Response(content=r.text, media_type="text/plain")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="backend unavailable")


@app.get("/metrics/triton")
async def metrics_triton():
    return await _proxy_metrics(TRITON_METRICS_URL)


@app.get("/metrics/vllm")
async def metrics_vllm():
    return await _proxy_metrics(VLLM_METRICS_URL)


@app.get("/metrics/node")
async def metrics_node():
    return await _proxy_metrics(NODE_METRICS_URL)


def _detect_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip().splitlines()
        if out:
            return out[0].strip()
    except Exception:
        pass
    return os.environ.get("GPU_NAME", "unknown GPU")


@app.get("/info")
async def info(_user: dict = Depends(get_current_user)):
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": _detect_gpu(),
        "version": "1.0",
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _infer_triton(batch: np.ndarray) -> np.ndarray:
    client = httpclient.InferenceServerClient(url=TRITON_URL, timeout=10.0)
    inp = httpclient.InferInput("INPUT__0", batch.shape, "FP32")
    inp.set_data_from_numpy(batch)
    out = httpclient.InferRequestedOutput("OUTPUT__0")
    result = client.infer(model_name=CV_MODEL, inputs=[inp], outputs=[out])
    return result.as_numpy("OUTPUT__0")


@app.post("/vision")
async def vision(
    file: UploadFile = File(...),
    _user: dict = Depends(get_current_user),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="not an image")

    t0 = time.perf_counter()
    try:
        batch = _preprocess(img)
        logits = await asyncio.to_thread(_infer_triton, batch)
    except Exception:
        raise HTTPException(status_code=502, detail="inference failed")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    probs = _softmax(logits[0])
    top5 = np.argsort(-probs)[:5]
    predictions = [
        {"class_id": int(i), "label": IMAGENET_CLASSES[int(i)], "score": float(probs[int(i)])}
        for i in top5
    ]
    return {"model": CV_MODEL, "predictions": predictions, "latency_ms": elapsed_ms}


@app.post("/chat")
async def chat(request: Request, _user: dict = Depends(get_current_user)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    if not isinstance(body, dict) or not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages required")

    stream = bool(body.get("stream", True))
    payload = dict(body)
    payload["model"] = LLM_MODEL
    payload["stream"] = stream

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = await client.send(
            client.build_request(
                "POST",
                f"{VLLM_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ),
            stream=stream,
        )
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(status_code=502, detail="vllm unavailable")

    if upstream.status_code >= 400:
        text = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        if 400 <= upstream.status_code < 500:
            raise HTTPException(status_code=400, detail=text.decode(errors="ignore"))
        raise HTTPException(status_code=502, detail="vllm error")

    if not stream:
        data = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(content=data, media_type="application/json")

    async def _iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(_iter(), media_type="text/event-stream")
