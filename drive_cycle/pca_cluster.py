"""TETDC Stage 6c — PCA + K-means clustering of the micro-trip bank
(third methodological pillar per master prompt section 4).

Standardises 8 kinematic features of 200,365 micro-trips, runs PCA
(variance structure + loadings) and K-means (k selected by silhouette over
4..8), profiles the resulting micro-trip classes and their hour-of-day mix.
Output: microtrip_pca_kmeans.json (+ centroids table csv)
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(BASE, "data", "processed")
FEATS = os.path.join(OUTD, "microtrips_features.parquet")

df = pd.read_parquet(FEATS)
cols = ["duration_s", "distance_m", "v_mean", "v_max", "a_max", "d_max",
        "P_idle", "a_rms"]
X = df[cols].to_numpy()
sc = StandardScaler().fit(X)
Z = sc.transform(X)

pca = PCA(random_state=SEED).fit(Z)
evr = pca.explained_variance_ratio_
print("PCA explained variance:", np.round(evr[:4], 3), "| n for 90%:",
      int(np.searchsorted(np.cumsum(evr), 0.90) + 1))
loadings = pd.DataFrame(pca.components_[:3].T, index=cols,
                        columns=[f"PC{i+1}" for i in range(3)]).round(3)
print(loadings)

sil = {}
for k in range(4, 9):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z)
    s = silhouette_score(Z, km.labels_, sample_size=20000, random_state=SEED)
    sil[k] = s
    print(f"  k={k}: silhouette {s:.3f}")
best_k = max(sil, key=sil.get)
km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(Z)
df["cluster"] = km.labels_

cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=cols)
cent["n"] = [(df["cluster"] == i).sum() for i in range(best_k)]
cent["share_pct"] = (cent["n"] / len(df) * 100).round(1)
cent = cent.sort_values("v_mean").reset_index(drop=True)
cent.to_csv(os.path.join(OUTD, "microtrip_clusters.csv"), index=False,
            encoding="utf-8-sig")
print(cent.round(2).to_string())

# hour-of-day cluster mix (UTC hours, consistent with stage 3)
mix = (df.groupby(["hour", "cluster"]).size().unstack(fill_value=0))
mix = mix.div(mix.sum(axis=1), axis=0).round(3)

# interpretive labels from centroid ordering by v_mean
labels = []
for i, row in cent.iterrows():
    if row["P_idle"] > 30 and row["v_mean"] < 4:
        lab = "idle-heavy / crawling"
    elif row["v_mean"] < 6:
        lab = "congested short-haul"
    elif row["v_mean"] < 9:
        lab = "mixed urban"
    elif row["v_max"] > 22:
        lab = "arterial free-flow"
    else:
        lab = "suburban steady"
    labels.append(lab)

out = {
    "seed": SEED, "n_microtrips": int(len(df)), "features": cols,
    "pca": {"explained_variance_ratio": [round(float(x), 4) for x in evr[:5]],
            "n_components_for_90pct": int(np.searchsorted(np.cumsum(evr), 0.90) + 1),
            "loadings_PC1_PC3": loadings.to_dict()},
    "kmeans": {"silhouette_by_k": {int(k): round(float(v), 3) for k, v in sil.items()},
               "selected_k": int(best_k),
               "cluster_labels": labels,
               "centroids": cent.round(3).to_dict(orient="records")},
    "hour_cluster_mix": {int(h): {int(c): float(mix.loc[h, c])
                                  for c in mix.columns} for h in mix.index},
}
with open(os.path.join(OUTD, "microtrip_pca_kmeans.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("saved microtrip_pca_kmeans.json + microtrip_clusters.csv")
