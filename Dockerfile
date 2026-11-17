# Base image: same minor version as the local venv (Python 3.13)
FROM python:3.13-slim

# Runtime shared library needed by numpy / scikit-learn (OpenMP runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install Python dependencies first (better layer caching)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# 2) Copy the FastAPI application
COPY app.py .

# 3) Copy the MLflow model registry (sqlite DB + artifacts) needed by app.py
#    app.py loads the champion via: mlflow.sklearn.load_model("models:/churn-model@champion")
COPY mlflow.db .
COPY mlruns ./mlruns

# Optional: keep the pickled model alongside (not used by the API, but handy)
COPY model.pkl .

# 4) Rewrite the Windows absolute paths stored in the registry DB to /app,
#    so the model can be found inside the Linux container.
COPY _fix_paths.py .
RUN python _fix_paths.py && rm _fix_paths.py

EXPOSE 8000

# Run the API with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
