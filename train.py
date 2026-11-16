from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "Metro_Interstate_Traffic_Volume.csv"
FLOW_FEATURES = [
    "hour",
    "day_of_week",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]
FLOW_BANDS = ["Low", "Moderate", "High", "Very high"]


df = pd.read_csv(DATASET_PATH)
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek


def estimate_speed(row):
    is_rush_hour = row["hour"] in [7, 8, 9, 16, 17, 18]
    if is_rush_hour and row["traffic_volume"] < 2500:
        return np.random.randint(5, 20)
    if is_rush_hour:
        return np.random.randint(25, 45)
    return np.random.randint(55, 80)


np.random.seed(42)
df["average_speed"] = df.apply(estimate_speed, axis=1)

cluster_features = ["hour", "day_of_week", "traffic_volume", "average_speed"]
cluster_data = df[cluster_features].dropna()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_data = cluster_data.copy()
cluster_data["cluster"] = kmeans.fit_predict(cluster_scaled)

cluster_mean_speeds = (
    cluster_data.groupby("cluster")["average_speed"]
    .mean()
    .sort_values(ascending=False)
)
state_labels = [
    "🟢 Free Flowing (High Speed)",
    "🟡 Moderate Traffic",
    "🟠 Heavy Traffic (Slow)",
    "🔴 Gridlock (Bumper-to-Bumper)",
]
cluster_to_state = {
    cluster_id: label
    for cluster_id, label in zip(cluster_mean_speeds.index, state_labels)
}

joblib.dump(kmeans, APP_DIR / "kmeans_traffic_model.pkl")
joblib.dump(scaler, APP_DIR / "scaler.pkl")
joblib.dump(cluster_to_state, APP_DIR / "cluster_mapping.pkl")


flow_data = df.sort_values("date_time").dropna(
    subset=FLOW_FEATURES + ["traffic_volume"]
)
flow_features = flow_data[FLOW_FEATURES]
flow_target = flow_data["traffic_volume"]
split_index = int(len(flow_data) * 0.8)
train_features = flow_features.iloc[:split_index]
test_features = flow_features.iloc[split_index:]
train_volume = flow_target.iloc[:split_index]
test_volume = flow_target.iloc[split_index:]

volume_thresholds = train_volume.quantile([0.25, 0.5, 0.75]).tolist()
volume_bins = [-np.inf, *volume_thresholds, np.inf]
train_bands = pd.cut(
    train_volume, bins=volume_bins, labels=FLOW_BANDS, include_lowest=True
).astype(str)
test_bands = pd.cut(
    test_volume, bins=volume_bins, labels=FLOW_BANDS, include_lowest=True
).astype(str)

flow_regressor = RandomForestRegressor(
    n_estimators=120,
    max_leaf_nodes=128,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)
flow_regressor.fit(train_features, train_volume)
volume_predictions = flow_regressor.predict(test_features)

classifiers = {
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_leaf_nodes=128,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    ),
}
classification_scores = []
for model_name, model in classifiers.items():
    model.fit(train_features, train_bands)
    predicted_bands = model.predict(test_features)
    classification_scores.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(test_bands, predicted_bands),
            "Macro F1": f1_score(
                test_bands, predicted_bands, average="macro", zero_division=0
            ),
        }
    )

feature_importances = (
    pd.DataFrame(
        {
            "Feature": FLOW_FEATURES,
            "Importance": classifiers["Random Forest"].feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

flow_analysis = {
    "features": FLOW_FEATURES,
    "volume_bands": FLOW_BANDS,
    "volume_thresholds": volume_thresholds,
    "regressor": flow_regressor,
    "classifiers": classifiers,
    "regression_metrics": {
        "MAE": mean_absolute_error(test_volume, volume_predictions),
        "RMSE": mean_squared_error(test_volume, volume_predictions) ** 0.5,
        "R2": r2_score(test_volume, volume_predictions),
    },
    "classification_metrics": classification_scores,
    "feature_importances": feature_importances,
    "train_rows": len(train_features),
    "test_rows": len(test_features),
    "test_start": str(flow_data["date_time"].iloc[split_index]),
    "test_end": str(flow_data["date_time"].iloc[-1]),
}
joblib.dump(flow_analysis, APP_DIR / "flow_analysis.pkl")

print("Traffic models trained.")
print(
    f"Volume regression holdout MAE: "
    f"{flow_analysis['regression_metrics']['MAE']:.1f} vehicles/hour"
)
print("Traffic-level classifier comparison:")
for score in classification_scores:
    print(
        f"  {score['Model']}: accuracy={score['Accuracy']:.3f}, "
        f"macro F1={score['Macro F1']:.3f}"
    )
print("Random Forest feature importance:")
print(feature_importances.to_string(index=False))
