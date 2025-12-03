# Deployment Guide — Phishing Classification Pipeline

This guide describes how to run the phishing classification API locally, build a container image, and deploy it in common environments. It assumes that a trained model has been persisted using the pipeline (`src/pipeline.py`) and that the repository is in the sanitized, enterprise-ready state.

---

## 1. Prerequisites

- Python 3.10+
- `pip` and a virtual environment tool (e.g., `venv`)
- Docker (for container-based deployment)
- A trained model saved to the configured `results` directory (see `config/config.yaml`)

---

## 2. Running the API Locally (Without Docker)

### 2.1 Create and Activate a Virtual Environment

```bash
cd /path/to/ml-engineering-full-cycle-pipeline-showcase

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Train and Persist a Model (One-Time or as Needed)

If you have not already trained and saved models:

```bash
python src/pipeline.py --config config/config.yaml
```

Ensure that `config/config.yaml` has `output.save_models: true` if you want the pipeline to persist models and metadata to the `results` directory.

### 2.4 Run the FastAPI Service

The API implementation lives in `api/serve.py` and uses FastAPI + uvicorn.

```bash
uvicorn api.serve:app --host 0.0.0.0 --port 8000
```

The service will:

- Load the model specified by the `MODEL_NAME` environment variable (default: `RandomForest`).
- Expose:
  - `GET /health` — liveness and model identity.
  - `POST /predict` — phishing classification endpoint.

---

## 3. Example Requests (curl)

### 3.1 Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Expected response:

```json
{
  "status": "ok",
  "model": "RandomForest"
}
```

### 3.2 Prediction Request

The `POST /predict` endpoint expects a JSON payload with a `features` object. Keys should match the feature names used at training time, and values should be numeric. In this reference implementation, features are mapped to a sorted list by key before being passed to the model.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "features": {
          "DomainAgeMonths": 24,
          "IsResponsive": 1,
          "LineOfCode": 800,
          "NoOfExternalRef": 10,
          "NoOfImage": 5,
          "NoOfPopup": 0,
          "NoOfSelfRef": 15,
          "NoOfURLRedirect": 1,
          "Robots": 1
        }
      }'
```

Example response:

```json
{
  "model_name": "RandomForest",
  "prediction": 0,
  "probability": 0.12
}
```

> Note: In a production deployment, the API would typically apply the persisted preprocessing pipeline (scaling, encoding) to raw domain features before calling the model. This reference implementation expects pre-aligned features.

---

## 4. Building and Running the Docker Image

### 4.1 Build the Image

From the repository root:

```bash
docker build -t phishing-ml-api:latest .
```

This uses the provided `Dockerfile`, which:

- Uses `python:3.10-slim` as a base image.
- Installs system dependencies needed for the scientific Python stack.
- Installs all Python dependencies from `requirements.txt`.
- Copies the application code into `/app`.
- Exposes port `8000`.
- Starts the FastAPI server with uvicorn.

### 4.2 Run the Container

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_NAME=RandomForest \
  phishing-ml-api:latest
```

If your model artifacts are not baked into the image, you can mount the `results` directory from the host:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_NAME=RandomForest \
  -v "$(pwd)/results:/app/results" \
  phishing-ml-api:latest
```

The API will then be accessible at `http://localhost:8000`.

---

## 5. Deployment Patterns

### 5.1 Local Development and Testing

- Recommended for:
  - Iterating on model and preprocessing strategy.
  - Verifying end-to-end behavior (pipeline + API).
- Run via:
  - `python src/pipeline.py` for training/evaluation.
  - `uvicorn api.serve:app` for serving.

### 5.2 Virtual Machine (VM) Deployment

Typical pattern:

1. Provision a Linux VM with Python 3.10 and Docker (optional).
2. Clone the repository and install dependencies (or deploy the Docker image).
3. Configure:
   - Environment variables (e.g., `MODEL_NAME`, log level).
   - System service (e.g., `systemd` unit) to run uvicorn on startup.
4. Expose port 8000 behind a load balancer or API gateway, with TLS termination.

### 5.3 Container-Oriented Platforms

The provided Dockerfile is compatible with container orchestrators such as:

- Kubernetes (EKS, AKS, GKE, self-managed).
- AWS ECS / Fargate.
- Azure Container Apps.
- Google Cloud Run.

Common steps:

1. Build and push the image to a container registry (ECR, ACR, GCR, etc.).
2. Define:
   - Deployment (or equivalent) referencing the image.
   - Service / Ingress / Gateway for external access.
   - ConfigMaps / Secrets for configuration and credentials.
3. Mount or inject model artifacts (e.g., from a persistent volume or object storage).
4. Implement health checks using `GET /health`.

### 5.4 Integration with MLOps Platforms

The API and pipeline are designed to integrate with:

- **Model registries**: Register trained models and promote specific versions to production.
- **Feature stores**: Ensure consistent feature computation between training and serving.
- **Monitoring and logging stacks**: Forward API logs and metrics into systems such as Prometheus, Datadog, or ELK for:
  - Latency and error rate monitoring.
  - Model performance monitoring (where linked to labeled feedback).

---

## 6. Operational Considerations

- **Security**:
  - Place the API behind an authenticated gateway or service mesh where appropriate.
  - Treat prediction requests as untrusted input (validate and sanitize features).
- **Performance**:
  - Benchmark throughput and latency under realistic load.
  - Scale horizontally using multiple replicas when deploying on orchestrators.
- **Reliability**:
  - Use readiness and liveness probes based on `GET /health`.
  - Consider blue-green or canary deployment strategies when promoting new models.

This guide is intentionally high-level but provides enough structure for a senior ML or platform engineer to adapt the reference implementation to their organization’s deployment stack.


