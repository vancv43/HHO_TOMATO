"""125-D color-statistical feature extractor for tomato ripeness/quality grading.

Feature layout (125 total), matching the manuscript's description
("color-statistical descriptors extracted from RGB, HSV, and Lab spaces,
channel histograms, and ripeness-sensitive color-ratio indices"):

  1. RGB channel stats:            3 channels x 4 stats (mean,std,skew,kurt) = 12
  2. HSV channel stats:            3 channels x 4 stats                     = 12
  3. Lab channel stats:            3 channels x 4 stats                     = 12
  4. 8-bin histograms:             9 channels x 8 bins                      = 72
  5. Ripeness/damage indices:      17 hand-designed indices                 = 17
  --------------------------------------------------------------------------
  Total                                                                    = 125
"""
import cv2
import numpy as np
from scipy.stats import skew, kurtosis

IMG_SIZE = 128
N_FEATURES = 125


def _stats4(channel):
    c = channel.astype(np.float64).ravel()
    return [c.mean(), c.std(), skew(c), kurtosis(c)]


def _hist8(channel, value_range):
    hist, _ = np.histogram(channel.ravel(), bins=8, range=value_range)
    hist = hist.astype(np.float64)
    s = hist.sum()
    return (hist / s if s > 0 else hist).tolist()


def extract_features(bgr_img):
    """bgr_img: HxWx3 uint8, as read by cv2.imread. Returns a 125-d float vector."""
    img = cv2.resize(bgr_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, Bl = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    feats = []
    # 1-3: channel stats (36)
    for ch in (R, G, B, H, S, V, L, A, Bl):
        feats += _stats4(ch)

    # 4: 8-bin histograms (72)
    feats += _hist8(R, (0, 256)); feats += _hist8(G, (0, 256)); feats += _hist8(B, (0, 256))
    feats += _hist8(H, (0, 180)); feats += _hist8(S, (0, 256)); feats += _hist8(V, (0, 256))
    feats += _hist8(L, (0, 256)); feats += _hist8(A, (0, 256)); feats += _hist8(Bl, (0, 256))

    # 5: ripeness / damage indices (17)
    Rf, Gf, Bf = R.astype(np.float64), G.astype(np.float64), B.astype(np.float64)
    eps = 1e-6
    total = Rf + Gf + Bf + eps
    idx = []
    idx.append(float((Rf / (Gf + eps)).mean()))                       # 1 red/green ratio
    idx.append(float((Rf / total).mean()))                             # 2 redness fraction
    idx.append(float((Gf / total).mean()))                             # 3 greenness fraction
    idx.append(float((Bf / total).mean()))                             # 4 blueness fraction
    idx.append(float(((Rf - Gf) / (Rf + Gf + eps)).mean()))            # 5 normalized red-green index
    hue = H.astype(np.float64)
    ripe_mask = (hue <= 10) | (hue >= 160)
    green_mask = (hue >= 35) & (hue <= 85)
    idx.append(float(ripe_mask.mean()))                                 # 6 ripe-hue pixel fraction
    idx.append(float(green_mask.mean()))                                # 7 unripe(green)-hue pixel fraction
    idx.append(float((V < 50).mean()))                                  # 8 dark-pixel fraction (shadow/damage)
    idx.append(float((V > 200).mean()))                                 # 9 bright-pixel fraction (blemish/highlight)
    idx.append(float(hue.std()))                                        # 10 hue std (ripening non-uniformity)
    edges = cv2.Canny(gray, 60, 150)
    idx.append(float((edges > 0).mean()))                               # 11 edge density (texture/damage)
    # local texture roughness via 5x5 local std
    mean_local = cv2.blur(gray.astype(np.float32), (5, 5))
    sq_local = cv2.blur((gray.astype(np.float32)) ** 2, (5, 5))
    local_var = np.clip(sq_local - mean_local ** 2, 0, None)
    idx.append(float(np.sqrt(local_var).mean()))                        # 12 mean local texture roughness
    hue_hist, _ = np.histogram(hue.ravel(), bins=18, range=(0, 180))
    p = hue_hist.astype(np.float64); p = p / (p.sum() + eps)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    idx.append(ent)                                                     # 13 hue histogram entropy
    idx.append(float(S.astype(np.float64).mean() / 255.0))              # 14 mean saturation (normalized)
    idx.append(float(A.astype(np.float64).mean()))                      # 15 Lab a* mean (green<->red axis)
    idx.append(float(Bl.astype(np.float64).mean()))                     # 16 Lab b* mean (blue<->yellow axis)
    idx.append(float((Rf > Gf).mean()))                                 # 17 fraction of red-dominant pixels
    feats += idx

    assert len(feats) == N_FEATURES, f"expected {N_FEATURES}, got {len(feats)}"
    return np.array(feats, dtype=np.float64)


FEATURE_NAMES = (
    [f"{c}_{s}" for c in ["R", "G", "B", "H", "S", "V", "L", "a", "b"] for s in ["mean", "std", "skew", "kurt"]]
    + [f"{c}_hist{b}" for c in ["R", "G", "B", "H", "S", "V", "L", "a", "b"] for b in range(8)]
    + ["rg_ratio", "redness_frac", "greenness_frac", "blueness_frac", "ngri",
       "ripe_hue_frac", "green_hue_frac", "dark_frac", "bright_frac", "hue_std",
       "edge_density", "local_texture", "hue_entropy", "sat_mean_norm", "lab_a_mean",
       "lab_b_mean", "red_dominant_frac"]
)
assert len(FEATURE_NAMES) == N_FEATURES
