import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

# =========================
# 1. Generate Dataset
# =========================
X, y = make_regression(n_samples=20000, n_features=10, noise=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2. Sampling Methods
# =========================

def random_sampling(X, y, ratio=0.3):
    n = len(X)
    k = int(ratio * n)
    indices = np.random.choice(np.arange(n), size=k, replace=False)
    return X[indices], y[indices]


def stratified_sampling(X, y, ratio=0.3, bins=10):
    y_binned = pd.qcut(y, q=bins, duplicates='drop')
    split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    for _, sample_idx in split.split(X, y_binned):
        return X[sample_idx], y[sample_idx]


def systematic_sampling(X, y, ratio=0.3):
    n = len(X)
    k = int(1 / ratio)
    indices = np.arange(0, n, k)
    return X[indices], y[indices]


def cluster_sampling(X, y, ratio=0.3, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    selected_indices = []
    samples_per_cluster = int((ratio * len(X)) / n_clusters)

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > samples_per_cluster:
            chosen = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
            selected_indices.extend(chosen)

    return X[selected_indices], y[selected_indices]

# =========================
# 3. Smoothing Methods
# =========================

def moving_average(X, window_size=3):
    X_smooth = np.copy(X)
    for i in range(X.shape[1]):
        X_smooth[:, i] = np.convolve(
            X[:, i],
            np.ones(window_size)/window_size,
            mode='same'
        )
    return X_smooth


def gaussian_smoothing(X, sigma=0.5):
    X_smooth = np.copy(X)
    for i in range(X.shape[1]):
        X_smooth[:, i] = gaussian_filter1d(X[:, i], sigma=sigma)
    return X_smooth

# =========================
# 4. Training Function
# =========================

def train_and_evaluate(X_train, y_train, X_test, y_test, max_iter=150):
    start = time.time()

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=max_iter)
    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return end - start, mse

# =========================
# 5. Run All Methods
# =========================

results = {}

# Baseline
results["Full Data"] = train_and_evaluate(X_train, y_train, X_test, y_test)

# Sampling Methods
X_rand, y_rand = random_sampling(X_train, y_train)
results["Random Sampling"] = train_and_evaluate(X_rand, y_rand, X_test, y_test, max_iter=50)

X_strat, y_strat = stratified_sampling(X_train, y_train)
results["Stratified Sampling"] = train_and_evaluate(X_strat, y_strat, X_test, y_test)

X_sys, y_sys = systematic_sampling(X_train, y_train)
results["Systematic Sampling"] = train_and_evaluate(X_sys, y_sys, X_test, y_test)

X_cluster, y_cluster = cluster_sampling(X_train, y_train)
results["Cluster Sampling"] = train_and_evaluate(X_cluster, y_cluster, X_test, y_test)

# Smoothing Methods (Full Data)
X_ma = moving_average(X_train)
results["Moving Average"] = train_and_evaluate(X_ma, y_train, X_test, y_test)

X_gauss = gaussian_smoothing(X_train)
results["Gaussian"] = train_and_evaluate(X_gauss, y_train, X_test, y_test)

# Hybrid Method
X_strat_ma = moving_average(X_strat)
results["Stratified + MA"] = train_and_evaluate(X_strat_ma, y_strat, X_test, y_test)

# =========================
# 6. Print Results
# =========================

print("\n===== FINAL COMPARISON =====\n")

for method, (t, mse) in results.items():
    print(f"{method:25s} → Time: {t:.4f} sec | MSE: {mse:.4f}")

# =========================
# 7. Plot Results
# =========================

methods = list(results.keys())
times = [results[m][0] for m in methods]
mses = [results[m][1] for m in methods]

plt.figure(figsize=(15,6))

# Training Time Plot
plt.subplot(1,2,1)
plt.bar(methods, times)
plt.xticks(rotation=45)
plt.title("Training Time Comparison")

# MSE Plot
plt.subplot(1,2,2)
plt.bar(methods, mses)
plt.xticks(rotation=45)
plt.title("MSE Comparison")

plt.tight_layout()
plt.show()
