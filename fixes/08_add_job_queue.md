# Step 08: Add Async Job Queue (Redis)

## 1. Objective

- Replace long **synchronous** `POST /v1/denoise` (Step 07) with **job submission**: `POST /v1/jobs` returns `job_id`; client polls `GET /v1/jobs/{id}` for status; result stored on disk or returned as URL.
- Use **Redis** as a broker and **RQ** (Redis Queue) or **Celery** — **RQ is simpler** for a first step.
- **Why:** Prevents HTTP timeouts, allows GPU worker scaling, matches production EO batch patterns.

## 2. Current Problem

- FastAPI worker blocks during ADMM iterations; concurrent requests contend for one GPU; no retry semantics.

## 3. Scope of Changes

### New dependencies

```
redis>=5.0.0
rq>=1.16.0
```

### New files

| Path | Purpose |
|------|---------|
| `workers/__init__.py` | Package |
| `workers/tasks.py` | `def run_denoise_job(job_id, input_path, output_path, kwargs): ...` |
| `api/jobs.py` | Router: create job, get status |
| `api/storage.py` | Paths under `data/jobs/{job_id}/input.png` and `output.png` |

### Modified files

| Path | Change |
|------|--------|
| `api/main.py` | Include `jobs` router; keep `/health`; deprecate or keep sync endpoint behind `SAR_SYNC_MODE=1` for backward compatibility |

### Infrastructure

- `docker-compose.yml` (optional but recommended): `redis:7-alpine` service.

## 4. Detailed Implementation Steps

1. **Add Redis**

   ```yaml
   # docker-compose.yml (snippet)
   services:
     redis:
       image: redis:7-alpine
       ports: ["6379:6379"]
   ```

2. **Implement `workers/tasks.py`**

   ```python
   from pathlib import Path
   from inference.service import SARDenoiseService
   import os
   import json

   def run_denoise_job(job_dir: str) -> None:
       job_dir = Path(job_dir)
       meta = json.loads((job_dir / "meta.json").read_text())
       inp = job_dir / "input.bin"  # or png path
       out = job_dir / "output.png"
       # load numpy from input — match how API saved it
       ...
       svc = SARDenoiseService(device=os.environ.get("SAR_DEVICE", "cpu"))
       svc.load_weights(meta["model_type"], Path(meta["checkpoint"]))
       ...
       (job_dir / "status.txt").write_text("done")
   ```

3. **API flow in `api/jobs.py`**

   - `POST /v1/jobs`: save upload to `data/jobs/<uuid>/input.png`, write `meta.json` with method, ADMM params, enqueue RQ job, return `{"job_id": "..."}`.
   - `GET /v1/jobs/{id}`: read `status.txt` or JSON `status.json`: `queued | running | done | failed`.
   - `GET /v1/jobs/{id}/result`: return PNG when done.

4. **RQ worker process**

   ```bash
   export REDIS_URL=redis://localhost:6379/0
   rq worker sar_denoise --url $REDIS_URL
   ```

5. **FastAPI lifespan** (optional): verify Redis connection on startup.

6. **Backward compatibility**

   - If env `SAR_USE_QUEUE=0`, keep Step 07 synchronous behavior.

## 5. Code-Level Guidance

### BEFORE

```python
@app.post("/v1/denoise")
async def denoise(...):
    # blocks
```

### AFTER

```python
from redis import Redis
from rq import Queue

redis_conn = Redis.from_url(os.environ["REDIS_URL"])
q = Queue("sar_denoise", connection=redis_conn)

@app.post("/v1/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    ...
    q.enqueue("workers.tasks.run_denoise_job", str(job_dir))
    return {"job_id": job_id}
```

**Note:** RQ needs import path `workers.tasks.run_denoise_job` discoverable from worker **working directory** (run worker from repo root).

## 6. Safety Constraints (VERY IMPORTANT)

- **MUST** retain synchronous `/v1/denoise` behind a flag **or** clearly version API (`/v1` sync vs `/v2/jobs`) so existing curl scripts keep working during migration.
- **MUST NOT** store secrets in Redis payloads; checkpoints paths only on trusted server.
- **Disk growth:** implement TTL cleanup for `data/jobs/*` (cron or periodic task) — document.

## 7. Testing & Verification

```bash
docker compose up -d redis
export REDIS_URL=redis://localhost:6379/0
export SAR_CHECKPOINT=...
rq worker sar_denoise &
uvicorn api.main:app --port 8000 &
JOB=$(curl -s -F "file=@small.png" http://127.0.0.1:8000/v1/jobs | jq -r .job_id)
curl -s http://127.0.0.1:8000/v1/jobs/$JOB
```

**Expected:** Status transitions to `done`; result downloadable.

## 8. Rollback Plan

- Set `SAR_USE_QUEUE=0`; disable router; stop worker; remove redis dependency if unused.

## 9. Result After This Step

- Non-blocking API suitable for Triton/batching (Step 09) and load tests in CI.
