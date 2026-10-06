import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris K-Means Clustering", page_icon="🌸")

# Cluster ID -> species (majority mapping; K-Means IDs are arbitrary).
# Fallback used if Iris.csv is not present next to app.py.
DEFAULT_MAP = {0: "Iris-versicolor", 1: "Iris-setosa", 2: "Iris-virginica"}


@st.cache_resource
def load_artifacts():
    with open("iris_kmeans_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("iris_kmeans_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features


@st.cache_data
def load_data():
    try:
        return pd.read_csv("Iris.csv")
    except FileNotFoundError:
        return None


model, features = load_artifacts()
df = load_data()

# Build cluster->species map from the data (majority species per cluster)
if df is not None:
    tmp = df.copy()
    tmp["cluster"] = model.predict(tmp[features])
    cluster_map = tmp.groupby("cluster")["Species"].agg(lambda s: s.mode()[0]).to_dict()
else:
    cluster_map = DEFAULT_MAP

st.title("🌸 Iris K-Means Clustering")
st.write("Flower ke measurements do, model batayega ki wo kis cluster / species se match karta hai.")

# ---- Sidebar inputs ----
st.sidebar.header("Flower Measurements (cm)")
ranges = {
    "SepalLengthCm": (4.0, 8.0, 5.8),
    "SepalWidthCm": (2.0, 4.5, 3.0),
    "PetalLengthCm": (1.0, 7.0, 4.3),
    "PetalWidthCm": (0.1, 2.5, 1.3),
}
values = {}
for feat in features:
    lo, hi, default = ranges.get(feat, (0.0, 10.0, 5.0))
    values[feat] = st.sidebar.slider(feat, lo, hi, default, step=0.1)

input_df = pd.DataFrame([values], columns=features)

st.subheader("Input")
st.dataframe(input_df, hide_index=True)

# ---- Prediction ----
if st.button("Predict Cluster", type="primary"):
    cluster = int(model.predict(input_df)[0])
    species = cluster_map.get(cluster, "Unknown")

    st.success(f"Cluster **{cluster}**  →  **{species}**")

    # Distance to each centroid (in scaled space)
    scaled = model.named_steps["scaler"].transform(input_df)
    dists = model.named_steps["kmeans"].transform(scaled)[0]
    dist_df = pd.DataFrame(
        {
            "Cluster": range(len(dists)),
            "Species": [cluster_map.get(i, "Unknown") for i in range(len(dists))],
            "Distance": dists.round(3),
        }
    )
    st.subheader("Distance from each cluster centre")
    st.dataframe(dist_df, hide_index=True)
    st.bar_chart(dist_df.set_index("Species")["Distance"])

# ---- Dataset preview ----
if df is not None:
    with st.expander("Dataset dekho"):
        st.dataframe(df.head(20), hide_index=True)