# Titrate ML Engineering and Deployment Workflow

Titrate includes a deployable deep-learning path without changing its scientific position: the Gaussian process remains the default small-data Bayesian-optimization model, while PyTorch is a scalable alternative for larger supervised datasets and production inference.

## 1. Inspect the GP vs PyTorch evidence

```bash
pip install -e ".[dev]"
streamlit run webapp/app.py
```

Open **Model Lab** in Streamlit's page navigation. The lab uses identical training rows and one untouched test split to compare:

- held-out RMSE, MAE, and R²;
- mean predictive uncertainty and empirical 95% interval coverage;
- nested learning curves showing error versus training-set size;
- PyTorch training/validation loss and early stopping; and
- interactive GP, PyTorch, and simulator predictions at the same operating condition.

The lab uses noiseless simulator values so it isolates surrogate approximation error. It does not replace the multi-seed BO benchmark, where the GP remains the acquisition model.

## 2. Train the PyTorch surrogate

```bash
pip install -e ".[dev,api,mlops]"
python experiments/train_torch_surrogate.py --samples 1500 --epochs 500
```

Outputs:

- `artifacts/torch_surrogate.pt` — serialized PyTorch model plus preprocessing metadata
- `results/torch_surrogate_metrics.json` — held-out RMSE, MAE, R², uncertainty and training metadata

The model uses an MLP with GELU activations and MC dropout. Inputs are scaled using physical process bounds and outputs are standardized during training. Early stopping uses a validation split.

## 3. Optional MLflow tracking

Point `MLFLOW_TRACKING_URI` at a local or hosted MLflow server before training:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python experiments/train_torch_surrogate.py
```

The training run logs parameters, held-out metrics, the model artifact and the metrics JSON under the `titrate-cstr-surrogate` experiment.

## 4. Serve the model locally

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

The API limits inference batches to 256 points and rejects malformed, non-finite, or out-of-bounds inputs. Integration tests train and load a real artifact before exercising health, metadata, prediction, and failure behavior.

## 5. Docker

The image trains a small deterministic model during the build, runs as a non-root user, and includes a container health check:

```bash
docker build -t titrate-api .
docker run -p 8000:8000 titrate-api
```

The container starts the FastAPI service with Uvicorn and is immediately healthy; no untracked local artifact is required. Override `MODEL_SAMPLES` and `MODEL_EPOCHS` as build arguments when needed. CI builds the image, starts it, and exercises all three API endpoints.

## 6. AWS App Runner deployment

The repository includes a real, manually triggered deployment workflow at `.github/workflows/deploy-aws.yml` and CloudFormation under `infra/aws/`. It uses GitHub OIDC, immutable commit-SHA image tags, ECR image scanning, App Runner health checks, and a post-deploy smoke test. No AWS access keys are stored in the repository.

### One-time AWS bootstrap

Run this from an AWS administrator session. If the account already has GitHub's OIDC provider, pass its ARN through `ExistingGitHubOidcProviderArn`.

```bash
aws cloudformation deploy \
  --stack-name titrate-github-oidc \
  --template-file infra/aws/github-oidc-role.yml \
  --capabilities CAPABILITY_NAMED_IAM
```

Then create a protected GitHub environment named `aws-production`:

1. Save the stack's `DeployRoleArn` output as the environment secret `AWS_DEPLOY_ROLE_ARN`.
2. Optionally set the environment variable `AWS_REGION` (default: `us-east-1`).
3. Add required reviewers to the environment if the account supports them.

### Deploy

Merge a green PR, then manually run **Deploy API to AWS App Runner** from `main`. The workflow:

1. exchanges GitHub's short-lived OIDC token for the scoped AWS role;
2. creates the `titrate-api` ECR repository if it does not exist;
3. builds and pushes an image tagged with the exact Git commit SHA;
4. deploys `infra/aws/apprunner.yml` through CloudFormation; and
5. waits for `/health` to return healthy.

The workflow deliberately refuses to deploy from non-`main` refs. The repository does not claim a live AWS deployment until this bootstrap is completed and the workflow succeeds.

For larger production models, keep the same image/API contract but fetch a separately versioned artifact from S3 at startup and add CloudWatch alarms around latency, error rate, and model-load failures.

## Why both GP and PyTorch?

Titrate's Gaussian process is still the better default when experiment budgets are only tens of samples because it provides strong small-data behavior and calibrated posterior uncertainty. The PyTorch surrogate is intentionally an additional model for larger datasets, model-comparison experiments, deep-learning experience and production serving.
