# Titrate ML Engineering Workflow

This branch extends Titrate from a Bayesian-optimization research codebase into a deployable ML system while keeping the Gaussian-process model as the small-data BO default.

## 1. Train the PyTorch surrogate

```bash
pip install -e ".[dev,api,mlops]"
python experiments/train_torch_surrogate.py --samples 1500 --epochs 500
```

Outputs:

- `artifacts/torch_surrogate.pt` — serialized PyTorch model plus preprocessing metadata
- `results/torch_surrogate_metrics.json` — held-out RMSE, MAE, R², uncertainty and training metadata

The model uses an MLP with GELU activations and MC dropout. Inputs are scaled using physical process bounds and outputs are standardized during training. Early stopping uses a validation split.

## 2. Optional MLflow tracking

Point `MLFLOW_TRACKING_URI` at a local or hosted MLflow server before training:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python experiments/train_torch_surrogate.py
```

The training run logs parameters, held-out metrics, the model artifact and the metrics JSON under the `titrate-cstr-surrogate` experiment.

## 3. Serve the model locally

```bash
TITRATE_MODEL_PATH=artifacts/torch_surrogate.pt \
uvicorn titrate.serving.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

Example request:

```json
{
  "points": [[360.0, 2.0, 1.0]],
  "mc_samples": 100
}
```

The response includes both the predicted objective and predictive standard deviation.

## 4. Docker

Train the artifact first, then:

```bash
docker build -t titrate-api .
docker run -p 8000:8000 titrate-api
```

The container starts the FastAPI service with Uvicorn.

## 5. AWS deployment path

A clean deployment path is:

1. Run tests in GitHub Actions.
2. Train/version the selected model artifact.
3. Build the Docker image.
4. Push the image to Amazon ECR.
5. Deploy the container with AWS App Runner or ECS/Fargate.
6. Store model artifacts in S3 for larger production models instead of baking them into the image.
7. Send application logs/metrics to CloudWatch.

The repository does not claim an AWS deployment until credentials and infrastructure are actually configured. The current implementation is container-ready for that step.

## Why both GP and PyTorch?

Titrate's Gaussian process is still the better default when experiment budgets are only tens of samples because it provides strong small-data behavior and calibrated posterior uncertainty. The PyTorch surrogate is intentionally an additional model for larger datasets, model-comparison experiments, deep-learning experience and production serving.
