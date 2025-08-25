# anomaly_methods.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# PyTorch for USAD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feature_utils import get_scaled_features_enhanced

# --- Device helper (MPS / CPU) ---
def get_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------------------
# K-MEANS anomaly scoring
# ----------------------------
def apply_kmeans_detection(df, config):
    try:
        km_config = config.get('kmeans', {})
        print(f" apply_kmeans_detection() : inputs - config is : {km_config}")
        
        n_clusters = km_config.get('n_clusters', 8)
        init = km_config.get('init', 'k-means++')
        n_init = km_config.get('n_init', 10)
        max_iter = km_config.get('max_iter', 300)
        contamination = km_config.get('contamination', 0.1)
        random_state = km_config.get('random_state', 42)
        
        # Use enhanced scaling method (consistent with DBSCAN)
        X_scaled, scaler, feature_names = get_scaled_features_enhanced(
            df, km_config.get('scaling_method', 'standard')
        )
        
        km = KMeans(n_clusters=n_clusters,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state)
        
        labels = km.fit_predict(X_scaled)
        centers = km.cluster_centers_
        
        # Calculate distances to cluster centers
        distances = np.linalg.norm(X_scaled - centers[labels], axis=1)
        
        # Use contamination to set threshold
        threshold = np.percentile(distances, (1 - contamination) * 100)
        anomalies = distances > threshold
        
        df['kmeans'] = anomalies
        df['kmeans_anomaly_scores'] = distances
        
        return anomalies, distances
        
    except Exception as e:
        print(f"Error in KMeans detection: {e}")
        return np.zeros(len(df), dtype=bool), np.zeros(len(df))


# ----------------------------
# USAD (PyTorch) implementation
# ----------------------------
class USADEncoder(nn.Module):
    def __init__(self, inp_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, max(32, latent_dim*2)),
            nn.ReLU(),
            nn.Linear(max(32, latent_dim*2), latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class USADDecoder(nn.Module):
    def __init__(self, inp_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, max(32, latent_dim*2)),
            nn.ReLU(),
            nn.Linear(max(32, latent_dim*2), inp_dim)
        )
    def forward(self, z):
        return self.net(z)

class USAD(nn.Module):
    """
    USAD style: two AEs (A->B and B->A), simplified to work with tabular / windowed multivariate inputs.
    We'll train to minimize joint reconstruction losses.
    """
    def __init__(self, inp_dim, latent_dim=64):
        super().__init__()
        self.E = USADEncoder(inp_dim, latent_dim)
        self.D1 = USADDecoder(inp_dim, latent_dim)
        self.D2 = USADDecoder(inp_dim, latent_dim)

    def forward(self, x):
        z = self.E(x)
        r1 = self.D1(z)
        z2 = self.E(r1)
        r2 = self.D2(z2)
        return r1, r2

def train_usad(X, latent_dim=32, epochs=30, batch_size=128, lr=1e-3, device=None, verbose=False):
    """
    X: numpy array (n_samples, n_features) or torch tensor
    Returns trained model and scaler.
    """
    device = device or get_torch_device()
    if isinstance(X, np.ndarray):
        X_np = X.astype(np.float32)
    else:
        X_np = X.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_np).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(Xs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = USAD(inp_dim=Xs.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean")

    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            r1, r2 = model(batch)
            loss1 = mse(r1, batch)
            loss2 = mse(r2, batch)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item()) * batch.size(0)
        epoch_loss /= len(loader.dataset)
        if verbose and (ep % max(1, epochs//5) == 0):
            print(f"[USAD] epoch {ep+1}/{epochs} loss={epoch_loss:.6f}")
    return model, scaler

def usad_score(model, scaler, X, device=None):
    """
    Compute per-row USAD anomaly score (reconstruction error).
    X: dataframe or numpy array
    Returns: numpy array scores
    """
    device = device or get_torch_device()
    if isinstance(X, pd.DataFrame):
        X_np = X.fillna(0).to_numpy(dtype=np.float32)
    else:
        X_np = X.astype(np.float32)
    Xs = scaler.transform(X_np).astype(np.float32)
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(Xs).to(device)
        r1, r2 = model(t)
        # We'll use mean square error between input and r2 (full loop) as final score
        err = ((r2 - t) ** 2).mean(dim=1).cpu().numpy()
    return err


# ----------------------------
# Meta-AAD (practical active loop)
# ----------------------------
def meta_aad_active_loop(df, feature_cols, initial_label_budget=20, pool_fraction=0.02, random_state=0):
    """
    A light-weight active anomaly detection loop that:
      1. uses simple unsupervised scoring (isolation/kmeans/usad) to propose candidates,
      2. queries a small budget (simulated label or via UI callback),
      3. trains a logistic meta-model that predicts anomaly probability from method scores.
    This function returns:
      - meta_scores: per-row probability (0..1) from the learned meta-model
      - meta_model: the fitted logistic model
      - queried_idx: indices that were sampled (for audit)
    Notes:
      - You should replace the `get_label` function with real user labeling (or use simulated labels).
      - This is *not* a full RL meta-learner, but a pragmatic on-ramp to Meta-AAD behavior.
    """
    rng = np.random.RandomState(random_state)
    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    n = len(df)
    # Step A: form simple unsupervised scorers (kmeans distances, feature zscore)
    kdist = run_kmeans_anomaly(df, feature_cols, n_clusters=4, random_state=random_state)
    zscore_feat = np.abs((df[feature_cols].fillna(0).iloc[:, 0] - df[feature_cols].fillna(0).iloc[:, 0].mean()) /
                         (df[feature_cols].fillna(0).iloc[:, 0].std() or 1)).to_numpy()
    base_scores = np.vstack([kdist, zscore_feat]).T
    # Step B: select pool for querying (top by base combined score)
    combined = base_scores.sum(axis=1)
    pool_size = max(1, int(n * pool_fraction))
    candidate_idx = np.argsort(-combined)[:pool_size]
    # Randomly sample initial_label_budget within candidates
    queried_idx = rng.choice(candidate_idx, size=min(initial_label_budget, len(candidate_idx)), replace=False)

    # --- Simulated labeling: replace this with real UI labeling callback ---
    # For now, if you have df['label'] available (1 anomalous, 0 normal), use it.
    if 'label' in df.columns:
        y_true = df['label'].to_numpy(dtype=int)
    else:
        # fallback: assume top combined are anomalies (simulated)
        y_true = np.zeros(n, dtype=int)
        y_true[combined.argsort()[-int(max(1, 0.02 * n)):]] = 1

    X_queried = base_scores[queried_idx]
    y_queried = y_true[queried_idx]

    # Train meta-model (logistic regressor) on queried examples
    if len(np.unique(y_queried)) <= 1:
        # insufficient diversity: create a weak prior (use combined scaled)
        meta_probs = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
        meta_model = None
    else:
        meta = LogisticRegression(solver="liblinear", random_state=random_state)
        meta.fit(X_queried, y_queried)
        meta_model = meta
        meta_probs = meta.predict_proba(base_scores)[:, 1]
    return meta_probs, meta_model, queried_idx

# EOF

