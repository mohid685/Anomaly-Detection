import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np

from config import BACKBONE, LAYERS, DEVICE, CORESET_RATIO, NUM_NEIGHBORS


class FeatureExtractor:
    """Frozen pretrained CNN, hooks into intermediate layers to get patch features."""

    def __init__(self):
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.model = wide_resnet50_2(weights=weights).to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.features = {}
        for layer_name in LAYERS:
            layer = dict(self.model.named_modules())[layer_name]
            layer.register_forward_hook(self._hook(layer_name))

    def _hook(self, name):
        def fn(_, __, output):
            self.features[name] = output
        return fn

    @torch.no_grad()
    def __call__(self, x):
        self.features = {}
        x = x.to(DEVICE)
        _ = self.model(x)

        layer_outputs = [self.features[l] for l in LAYERS]
        target_size = layer_outputs[0].shape[-2:]
        resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                   for f in layer_outputs]
        embedding = torch.cat(resized, dim=1)  # (B, C, H, W)
        return embedding


def embedding_to_patches(embedding):
    """(B, C, H, W) -> (B*H*W, C) patch-level feature vectors."""
    b, c, h, w = embedding.shape
    patches = embedding.permute(0, 2, 3, 1).reshape(-1, c)
    return patches.cpu().numpy(), (b, h, w)


def coreset_subsample(patch_features, ratio=CORESET_RATIO, seed=42):
    """
    Greedy k-center coreset selection to shrink the memory bank while
    preserving coverage of the feature space. Falls back to random
    sampling of a candidate pool if the dataset is very large (for speed),
    and applies the ratio relative to that pool (not the raw total) so
    compression stays consistent regardless of dataset size.
    """
    n_total = patch_features.shape[0]
    rng = np.random.default_rng(seed)

    if n_total > 20000:
        idx_pool = rng.choice(n_total, 20000, replace=False)
        pool = patch_features[idx_pool]
    else:
        idx_pool = np.arange(n_total)
        pool = patch_features

    n_select = max(1, int(len(pool) * ratio))

    selected = [rng.integers(0, len(pool))]
    min_dists = np.linalg.norm(pool - pool[selected[0]], axis=1)

    n_iterations = min(n_select, len(pool)) - 1
    for _ in tqdm(range(n_iterations), desc="Coreset selection"):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = np.linalg.norm(pool - pool[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    selected_global_idx = idx_pool[selected]
    return patch_features[selected_global_idx]


class PatchCore:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.memory_bank = None
        self.knn = None
        self.feature_map_hw = None
        self.pos_mean = None
        self.pos_std = None

    def fit(self, train_loader):
        all_patches = []
        for batch in tqdm(train_loader, desc="Extracting features"):
            embedding = self.extractor(batch)
            patches, (b, h, w) = embedding_to_patches(embedding)
            self.feature_map_hw = (h, w)
            all_patches.append(patches)

        all_patches = np.concatenate(all_patches, axis=0)
        print(f"Total training patches: {all_patches.shape[0]}")

        self.memory_bank = coreset_subsample(all_patches)
        print(f"Memory bank size after coreset subsampling: {self.memory_bank.shape[0]}")

        self.knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, algorithm="auto", n_jobs=-1)
        self.knn.fit(self.memory_bank)

    def calibrate(self, train_loader):
        """
        Computes per-spatial-location mean/std of raw anomaly scores over the
        normal training set. Some positions (e.g. a screw's tip or thread edge)
        score elevated in every image due to viewpoint/edge effects, not actual
        defects. This lets score() judge each position relative to its own
        expected normal range, instead of one flat threshold for the whole image.
        Only used when a pipeline explicitly opts in via use_position_norm.
        """
        maps = []
        for batch in tqdm(train_loader, desc="Calibrating position stats"):
            for i in range(batch.shape[0]):
                _, anomaly_map = self.score(batch[i:i + 1], use_position_norm=False)
                maps.append(anomaly_map)
        maps = np.stack(maps, axis=0)
        self.pos_mean = maps.mean(axis=0)
        self.pos_std = maps.std(axis=0) + 1e-6
        print(f"Calibration complete over {len(maps)} training images.")

    def score(self, x, top_k=5, subtract_baseline=False, use_position_norm=False):
        embedding = self.extractor(x)
        patches, (b, h, w) = embedding_to_patches(embedding)

        distances, _ = self.knn.kneighbors(patches)
        patch_scores = distances.mean(axis=1)
        anomaly_map = patch_scores.reshape(h, w)

        scoring_map = anomaly_map
        if use_position_norm and self.pos_mean is not None:
            scoring_map = (anomaly_map - self.pos_mean) / self.pos_std

        flat_scores = scoring_map.flatten()
        k = min(top_k, flat_scores.size)
        top_k_scores = np.sort(flat_scores)[-k:]
        image_score = top_k_scores.mean()

        if subtract_baseline:
            image_score = image_score - np.median(flat_scores)

        # anomaly_map returned is always the RAW (non-normalized) map, so pixel-level
        # AUROC and heatmap visualizations are unaffected by position normalization —
        # only the single scalar image_score is affected.
        return image_score, anomaly_map

    def save(self, path):
        torch.save({
            "memory_bank": self.memory_bank,
            "feature_map_hw": self.feature_map_hw,
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
        }, path)

    def load(self, path):
        data = torch.load(path, weights_only=False)
        self.memory_bank = data["memory_bank"]
        self.feature_map_hw = data["feature_map_hw"]
        self.pos_mean = data.get("pos_mean")
        self.pos_std = data.get("pos_std")
        self.knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, algorithm="auto", n_jobs=-1)
        self.knn.fit(self.memory_bank)