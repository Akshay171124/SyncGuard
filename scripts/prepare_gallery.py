#!/usr/bin/env python
"""Bundle precomputed FakeAVCeleb features into demo gallery .npz files.

For each clip in the gallery manifest, copy the raw .mp4 and load the
existing {mouth_crops.npy, audio.wav, ear_features.npy} from the processed
features directory, then bundle them into demo_assets/gallery/<clip_id>.npz.
"""
import argparse
import json
import shutil
import wave
from pathlib import Path

import numpy as np


def load_wav(path):
    """Read a .wav file to (samples_array, sample_rate) — stdlib, no soundfile."""
    with wave.open(str(path), "rb") as w:
        n_frames = w.getnframes()
        sr = w.getframerate()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n_frames)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    if w.getnchannels() > 1:
        audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    updated_clips = []
    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        processed = repo_root / clip["processed_dir"]
        video_src = repo_root / clip["video_path"]

        mouth_src = processed / "mouth_crops.npy"
        audio_src = processed / "audio.wav"
        ear_src = processed / "ear_features.npy"

        missing = [p for p in [video_src, mouth_src, audio_src, ear_src]
                   if not p.exists()]
        if missing:
            print(f"[skip] {clip_id}: missing {missing}")
            continue

        # Copy raw mp4
        video_dst = out_dir / f"{clip_id}.mp4"
        shutil.copy(video_src, video_dst)

        mouth_crops = np.load(mouth_src)
        audio, sr = load_wav(audio_src)
        ear = np.load(ear_src)

        np.savez(
            out_dir / f"{clip_id}.npz",
            mouth_crops=mouth_crops.astype(np.float32),
            audio_waveform=audio.astype(np.float32),
            ear_features=ear.astype(np.float32),
            sample_rate=np.int32(sr),
            duration_s=np.float32(clip.get("audio_duration_s", len(audio) / sr)),
        )
        print(f"[ok] {clip_id} [{clip['category']}] "
              f"crops={mouth_crops.shape} audio={audio.shape} ear={ear.shape}")

        updated_clips.append({
            "clip_id": clip_id,
            "sample_id": clip["sample_id"],
            "category": clip["category"],
            "ground_truth": clip["ground_truth"],
            "hpc_confidence": clip["hpc_confidence"],
            "audio_duration_s": clip.get("audio_duration_s", 0.0),
            "speaker_id": clip.get("speaker_id", ""),
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"clips": updated_clips}, f, indent=2)
    print(f"\nBundled {len(updated_clips)} clips into {out_dir}")


if __name__ == "__main__":
    main()
