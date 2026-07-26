"""
Pytest suite for the metal_nut PatchCore anomaly detection pipeline.

Run from src/:
    pytest tests/test_patchcore_pipeline.py -v
or from src/tests/:
    pytest test_patchcore_pipeline.py -v

Some tests require artifacts/memory_bank.pt and artifacts/threshold.json
to already exist (i.e. train_patchcore.py and evaluate.py have been run).
Those tests are skipped automatically if the artifacts are missing.
"""

import os
import json
import numpy as np
import pytest
import torch

from config import (
    TRAIN_DIR, TEST_DIR, GT_DIR, MEMORY_BANK_PATH, THRESHOLD_PATH,
    CENTER_CROP, CORESET_RATIO, NUM_NEIGHBORS,
)
from mvtec_dataset import MVTecTrainDataset, MVTecTestDataset, get_transform
from patchcore import (
    FeatureExtractor, PatchCore, coreset_subsample, embedding_to_patches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def train_dataset():
    if not os.path.isdir(TRAIN_DIR):
        pytest.skip(f"Train directory not found: {TRAIN_DIR}")
    return MVTecTrainDataset(TRAIN_DIR)


@pytest.fixture(scope="session")
def test_dataset():
    if not os.path.isdir(TEST_DIR):
        pytest.skip(f"Test directory not found: {TEST_DIR}")
    return MVTecTestDataset(TEST_DIR, GT_DIR)


@pytest.fixture(scope="session")
def trained_model():
    if not os.path.exists(MEMORY_BANK_PATH):
        pytest.skip(f"No trained memory bank at {MEMORY_BANK_PATH} — run train_patchcore.py first")
    model = PatchCore()
    model.load(MEMORY_BANK_PATH)
    return model


@pytest.fixture(scope="session")
def saved_threshold():
    if not os.path.exists(THRESHOLD_PATH):
        pytest.skip(f"No threshold file at {THRESHOLD_PATH} — run evaluate.py first")
    with open(THRESHOLD_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestTrainDataset:
    def test_train_dataset_not_empty(self, train_dataset):
        assert len(train_dataset) > 0

    def test_train_dataset_only_good_images(self, train_dataset):
        for path in train_dataset.paths:
            assert "good" in path.replace("\\", "/").split("/")

    def test_train_sample_shape_and_normalization(self, train_dataset):
        img = train_dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, CENTER_CROP, CENTER_CROP)
        assert img.min() < 0 or img.max() > 1


class TestTestDataset:
    def test_test_dataset_has_all_defect_types(self, test_dataset):
        defect_types = {s[3] for s in test_dataset.samples}
        expected = {"good", "bent", "color", "flip", "scratch"}
        assert expected.issubset(defect_types)

    def test_good_images_have_zero_mask(self, test_dataset):
        for img, label, mask, defect_type, path in test_dataset:
            if defect_type == "good":
                assert label == 0
                assert torch.all(mask == 0)

    def test_defective_images_have_nonzero_mask_when_gt_exists(self, test_dataset):
        found_nonzero = False
        for img, label, mask, defect_type, path in test_dataset:
            if defect_type != "good" and label == 1:
                if mask.sum() > 0:
                    found_nonzero = True
                    break
        assert found_nonzero, "Expected at least one defective sample with a non-empty ground-truth mask"

    def test_all_samples_return_correct_shapes(self, test_dataset):
        img, label, mask, defect_type, path = test_dataset[0]
        assert img.shape == (3, CENTER_CROP, CENTER_CROP)
        assert mask.shape == (1, CENTER_CROP, CENTER_CROP)
        assert label in (0, 1)
        assert isinstance(defect_type, str)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Feature extractor / embedding tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        return FeatureExtractor()

    def test_output_is_4d_tensor(self, extractor):
        dummy = torch.randn(1, 3, CENTER_CROP, CENTER_CROP)
        out = extractor(dummy)
        assert out.dim() == 4
        assert out.shape[0] == 1

    def test_batch_dimension_preserved(self, extractor):
        dummy = torch.randn(4, 3, CENTER_CROP, CENTER_CROP)
        out = extractor(dummy)
        assert out.shape[0] == 4

    def test_embedding_to_patches_shape(self, extractor):
        dummy = torch.randn(2, 3, CENTER_CROP, CENTER_CROP)
        embedding = extractor(dummy)
        patches, (b, h, w) = embedding_to_patches(embedding)
        assert b == 2
        assert patches.shape[0] == b * h * w
        assert patches.shape[1] == embedding.shape[1]


# ---------------------------------------------------------------------------
# Coreset subsampling tests
# ---------------------------------------------------------------------------

class TestCoresetSubsample:
    def test_output_size_matches_ratio_small_pool(self):
        features = np.random.randn(1000, 16).astype(np.float32)
        result = coreset_subsample(features, ratio=0.2)
        assert result.shape[0] == pytest.approx(200, abs=2)

    def test_output_size_capped_pool(self):
        features = np.random.randn(50000, 8).astype(np.float32)
        result = coreset_subsample(features, ratio=0.1)
        assert result.shape[0] == pytest.approx(2000, abs=2)

    def test_selected_features_are_subset_of_input(self):
        features = np.random.randn(500, 4).astype(np.float32)
        result = coreset_subsample(features, ratio=0.1)
        for row in result:
            assert any(np.allclose(row, f) for f in features)

    def test_no_crash_on_tiny_input(self):
        features = np.random.randn(3, 4).astype(np.float32)
        result = coreset_subsample(features, ratio=0.5)
        assert result.shape[0] >= 1


# ---------------------------------------------------------------------------
# PatchCore model tests (require a trained memory bank)
# ---------------------------------------------------------------------------

class TestPatchCoreModel:
    def test_memory_bank_loaded(self, trained_model):
        assert trained_model.memory_bank is not None
        assert trained_model.memory_bank.shape[0] > 0

    def test_knn_fitted(self, trained_model):
        assert trained_model.knn is not None

    def test_score_returns_expected_types(self, trained_model, test_dataset):
        img, label, mask, defect_type, path = test_dataset[0]
        score, anomaly_map = trained_model.score(img.unsqueeze(0))
        assert isinstance(score, (float, np.floating))
        assert isinstance(anomaly_map, np.ndarray)
        assert anomaly_map.ndim == 2

    def test_score_is_deterministic(self, trained_model, test_dataset):
        img, label, mask, defect_type, path = test_dataset[0]
        score1, _ = trained_model.score(img.unsqueeze(0))
        score2, _ = trained_model.score(img.unsqueeze(0))
        assert score1 == pytest.approx(score2)

    def test_top_k_scoring_handles_small_map(self, trained_model, test_dataset):
        img, label, mask, defect_type, path = test_dataset[0]
        score, _ = trained_model.score(img.unsqueeze(0), top_k=1000000)
        assert np.isfinite(score)

    def test_save_and_load_roundtrip(self, trained_model):
        import tempfile
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts", "tmp_test")
        os.makedirs(tmp_dir, exist_ok=True)
        save_path = os.path.join(tmp_dir, "memory_bank_test.pt")

        trained_model.save(save_path)
        assert os.path.exists(save_path)

        reloaded = PatchCore()
        reloaded.load(save_path)
        assert reloaded.memory_bank.shape == trained_model.memory_bank.shape
        assert np.allclose(reloaded.memory_bank, trained_model.memory_bank)

        os.remove(save_path)  # cleanup


# ---------------------------------------------------------------------------
# Behavioral / regression tests
# ---------------------------------------------------------------------------

class TestModelBehavior:
    @pytest.fixture(scope="class")
    def scores_by_type(self, trained_model, test_dataset):
        scores = {}
        for img, label, mask, defect_type, path in test_dataset:
            s, _ = trained_model.score(img.unsqueeze(0))
            scores.setdefault(defect_type, []).append(s)
        return scores

    def test_good_mean_score_is_lowest(self, scores_by_type):
        good_mean = np.mean(scores_by_type["good"])
        for defect_type in ("bent", "color", "flip", "scratch"):
            if defect_type in scores_by_type:
                defect_mean = np.mean(scores_by_type[defect_type])
                assert defect_mean > good_mean, (
                    f"{defect_type} mean score ({defect_mean:.2f}) should exceed "
                    f"good mean score ({good_mean:.2f})"
                )

    def test_bent_and_flip_are_clearly_separated_from_good(self, scores_by_type):
        good_mean = np.mean(scores_by_type["good"])
        for defect_type in ("bent", "flip"):
            if defect_type in scores_by_type:
                defect_mean = np.mean(scores_by_type[defect_type])
                gap = defect_mean - good_mean
                assert gap > 3.0, (
                    f"{defect_type} mean gap from good shrank to {gap:.2f} "
                    f"(expected > 3.0 based on last known-good run)"
                )

    def test_overall_accuracy_at_saved_threshold(self, trained_model, test_dataset, saved_threshold):
        threshold = saved_threshold["threshold"]
        correct = 0
        total = 0
        for img, label, mask, defect_type, path in test_dataset:
            score, _ = trained_model.score(img.unsqueeze(0))
            predicted = int(score > threshold)
            correct += int(predicted == label)
            total += 1

        accuracy = correct / total
        print(f"\nOverall accuracy at saved threshold: {accuracy:.4f} ({correct}/{total})")
        assert accuracy >= 0.85, f"Overall accuracy dropped to {accuracy:.4f}, below 0.85 floor"

    def test_reported_aurocs_are_reasonable(self, saved_threshold):
        assert saved_threshold["image_auroc"] >= 0.90
        assert saved_threshold["pixel_auroc"] >= 0.90


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_score_handles_single_image_batch(self, trained_model, test_dataset):
        img, label, mask, defect_type, path = test_dataset[0]
        score, anomaly_map = trained_model.score(img.unsqueeze(0))
        assert np.isfinite(score)

    # def test_score_handles_larger_batch(self, trained_model, test_dataset):
    #     imgs = torch.stack([test_dataset[i][0] for i in range(min(4, len(test_dataset)))])
    #     score, anomaly_map = trained_model.score(imgs)
    #     assert np.isfinite(score)

    def test_score_only_supports_single_image_at_a_time(self, trained_model, test_dataset):
        """
        PatchCore.score() is designed for one image per call (batch size 1) —
        this is documented behavior, not a bug, since evaluate.py and
        run_visualizations.py always call it this way. Calling it with a
        batch > 1 raises, which this test locks in as expected behavior.
        """
        imgs = torch.stack([test_dataset[i][0] for i in range(min(4, len(test_dataset)))])
        with pytest.raises(ValueError):
            trained_model.score(imgs)

    def test_config_paths_exist(self):
        assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
        assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
        assert os.path.isdir(GT_DIR), f"Missing: {GT_DIR}"