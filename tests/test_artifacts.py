"""Artifact loading: local files, HF Hub fallback (mocked), checksum verification."""
import shutil
from pathlib import Path

import pytest

from app import artifacts as art_mod
from app.settings import Settings
from conftest import ARTIFACTS, requires_weights


def make_settings(tmp: Path, repo=None, token=None) -> Settings:
    return Settings(artifact_dir=tmp, hf_model_repo=repo, hf_model_revision="main",
                    hf_token=token, max_upload_mb=10, torch_num_threads=None)


def copy_without_weights(dst: Path) -> None:
    for p in ARTIFACTS.iterdir():
        if p.suffix != ".safetensors":
            shutil.copy(p, dst / p.name)


@requires_weights
def test_missing_weights_without_repo_fails(tmp_path):
    copy_without_weights(tmp_path)
    with pytest.raises(FileNotFoundError, match="HF_MODEL_REPO is not set"):
        art_mod.load_artifacts(make_settings(tmp_path))


@requires_weights
def test_missing_weights_downloaded_from_hub(tmp_path, monkeypatch):
    copy_without_weights(tmp_path)
    calls = []

    def fake_download(repo_id, filename, revision, token):
        calls.append((repo_id, filename, revision, token))
        return str(ARTIFACTS / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded = art_mod.load_artifacts(make_settings(tmp_path, repo="someone/model", token="hf_secret"))
    assert calls == [("someone/model", "swin_tiny_breakhis.safetensors", "main", "hf_secret")]
    assert loaded.weights_path == ARTIFACTS / "swin_tiny_breakhis.safetensors"


@requires_weights
def test_checksum_mismatch_is_rejected(tmp_path):
    copy_without_weights(tmp_path)
    (tmp_path / "swin_tiny_breakhis.safetensors").write_bytes(b"not the real weights")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        art_mod.load_artifacts(make_settings(tmp_path))
