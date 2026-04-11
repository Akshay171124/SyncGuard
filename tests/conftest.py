"""Shared fixtures for SyncGuard test suite."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def default_config():
    """Minimal config dict matching configs/default.yaml structure."""
    return {
        "data": {
            "features_dir": "data/processed",
            "fakeavceleb_dir": "data/raw/FakeAVCeleb",
        },
        "preprocessing": {
            "video": {"fps": 25, "mouth_crop_size": 96},
            "audio": {"sample_rate": 16000, "target_fps": 49},
        },
        "model": {
            "visual_encoder": {
                "name": "av_hubert",
                "embedding_dim": 256,
                "freeze_pretrained": False,
            },
            "audio_encoder": {
                "name": "wav2vec2",
                "model_id": "facebook/wav2vec2-base-960h",
                "layer": 9,
                "embedding_dim": 256,
                "freeze_pretrained": True,
            },
            "classifier": {
                "name": "bilstm",
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "use_ear": False,
            },
            "audio_head": False,
            "cross_attention": {"enabled": False},
            "dct_extractor": {"enabled": False},
        },
        "training": {
            "pretrain": {
                "batch_size": 4,
                "lr": 1e-4,
                "moco_queue_size": 128,
                "temperature": 0.07,
                "temperature_range": [0.01, 0.5],
                "cross_modal_prediction": True,
                "cmp_weight": 0.5,
                "cmp_mask_ratio": 0.3,
            },
            "finetune": {
                "batch_size": 4,
                "lr": 5e-5,
                "gamma": 0.5,
                "delta": 1.0,
            },
        },
        "hardware": {
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
        },
    }


@pytest.fixture
def batch_dims():
    """Standard batch dimensions for tests."""
    return {"B": 4, "T": 20, "D": 256}


@pytest.fixture
def dummy_embeddings(batch_dims):
    """L2-normalized visual and audio embeddings."""
    B, T, D = batch_dims["B"], batch_dims["T"], batch_dims["D"]
    v = F.normalize(torch.randn(B, T, D), dim=-1)
    a = F.normalize(torch.randn(B, T, D), dim=-1)
    return v, a


@pytest.fixture
def dummy_mask(batch_dims):
    """Boolean mask simulating variable-length sequences."""
    B, T = batch_dims["B"], batch_dims["T"]
    mask = torch.ones(B, T, dtype=torch.bool)
    # Last sample has padding
    mask[-1, T // 2 :] = False
    return mask


@pytest.fixture
def dummy_labels(batch_dims):
    """Binary labels: half real, half fake."""
    B = batch_dims["B"]
    labels = torch.zeros(B, dtype=torch.long)
    labels[B // 2 :] = 1
    return labels
