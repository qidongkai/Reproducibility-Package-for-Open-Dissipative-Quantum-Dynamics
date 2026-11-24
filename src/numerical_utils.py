import numpy as np
import os
import pickle
from pathlib import Path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_cache(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def load_cache(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def relative_error(a, b):
    return np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-14)

def symmetric_kl(p, q):
    p = np.clip(p, 1e-14, None)
    q = np.clip(q, 1e-14, None)
    return 0.5 * np.sum(p * np.log(p / q) + q * np.log(q / p))