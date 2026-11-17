
# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# ===== 2. Connexion au serveur MLflow Docker =====
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")

# ===== 3. Lire et préparer les données =====
df = pd.read_csv("churn_data.csv")

# Séparer les variables explicatives et la cible
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["abonnement"]
)

y = df["churn"]

# ===== 4. Entraîner le modèle =====
modele = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modele.fit(X, y)

# ===== 5. Enregistrer le modèle dans MLflow =====
with mlflow.start_run() as run:

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.sklearn.log_model(
        sk_model=modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle"
    )

    print("Run ID :", run.info.run_id)

# ===== 6. Créer l'alias champion =====
client = MlflowClient()

versions = client.search_model_versions(
    "name='churn-model'"
)

# Récupérer la version créée par ce run
version = next(
    v for v in versions
    if v.run_id == run.info.run_id
)

client.set_registered_model_alias(
    "churn-model",
    "champion",
    version.version
)

print(
    f"Version {version.version} de churn-model "
    "enregistrée avec l'alias champion"
)
