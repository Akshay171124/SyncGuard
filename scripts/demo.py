#!/usr/bin/env python
"""SyncGuard research-showcase demo (Gradio).

Tab 1: curated gallery — click a clip to analyze.
Tab 2: live challenge (only if SYNCGUARD_LIVE=1) — webcam + audio swap.

Runs entirely on the Mac. No HPC, no internet required at demo time.
"""
import os
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo.inference import DemoInference
from src.demo.gallery import load_gallery, GalleryClip
from src.demo.explain import generate_explanation, SYNC_THRESHOLD


CONFIG = "configs/default.yaml"
CHECKPOINT = "demo_assets/checkpoints/finetune_best.pt"
AUDIO_CHECKPOINT = "demo_assets/checkpoints/audio_clf_best.pt"
GALLERY_DIR = "demo_assets/gallery"
LIVE_ENABLED = os.environ.get("SYNCGUARD_LIVE", "0") == "1"

C_REAL = "#27AE60"
C_FAKE = "#E74C3C"
C_THRESHOLD = "#95A5A6"


def render_sync_plot(sync_curve, threshold, dip_segments, fps, title=""):
    t = np.arange(len(sync_curve)) / fps
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    ax.plot(t, sync_curve, color="#1A5276", linewidth=2, label="s(t)")
    ax.axhline(threshold, color=C_THRESHOLD, linestyle="--",
               label=f"threshold {threshold:.2f}")
    for s, e in dip_segments:
        ax.axvspan(s, e, alpha=0.25, color=C_FAKE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sync score")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def verdict_banner_html(verdict, confidence, ground_truth):
    color = C_FAKE if verdict == "fake" else C_REAL
    icon = "FAKE" if verdict == "fake" else "REAL"
    correct = "Correct" if verdict == ground_truth else "Incorrect"
    badge_color = C_REAL if verdict == ground_truth else C_FAKE
    return f"""
    <div style="padding:16px; border-radius:12px; background:{color};
                color:white; font-size:22px; text-align:center;
                font-weight:600; margin-bottom:12px;">
        {icon} — {int(round(confidence*100))}% confidence
    </div>
    <div style="padding:8px 16px; border-radius:8px; background:#f4f4f4;
                color:{badge_color}; font-weight:600;">
        {correct} — ground truth: {ground_truth.upper()}
    </div>
    """


def build_demo():
    print("Loading SyncGuard (sync + cascade audio)...")
    infer = DemoInference(CONFIG, CHECKPOINT,
                          audio_checkpoint_path=AUDIO_CHECKPOINT)
    print(f"  device={infer.device}")

    print("Loading gallery...")
    clips: list[GalleryClip] = load_gallery(GALLERY_DIR)
    clip_by_id = {c.clip_id: c for c in clips}
    print(f"  {len(clips)} clips loaded")

    clip_choices = [
        (f"{c.clip_id}  [{c.category}]", c.clip_id) for c in clips
    ]

    def on_analyze(clip_id):
        if clip_id is None:
            return None, "", "", None
        clip = clip_by_id[clip_id]
        result = infer.analyze(
            mouth_crops=clip.mouth_crops,
            audio_waveform=clip.audio_waveform,
            ear_features=clip.ear_features,
            clip_duration_s=clip.duration_s,
        )
        plot = render_sync_plot(
            result.sync_curve, SYNC_THRESHOLD,
            result.sync_dip_segments, infer.fps_sync,
            title=f"Sync score over time — {clip.clip_id}",
        )
        banner = verdict_banner_html(
            result.verdict, result.confidence, clip.ground_truth,
        )
        explanation = generate_explanation(result)
        return str(clip.video_path), banner, explanation, plot

    with gr.Blocks(title="SyncGuard — Audio-Visual Deepfake Detection") as app:
        gr.Markdown(
            "# SyncGuard\n"
            "**Contrastive Audio-Visual Deepfake Detection**\n\n"
            "Pick a clip below, then click *Analyze* to see the model's "
            "verdict, the sync-score curve, and a natural-language "
            "explanation of *why*."
        )
        with gr.Tabs():
            with gr.Tab("Gallery"):
                with gr.Row():
                    clip_picker = gr.Radio(
                        choices=clip_choices,
                        label="Curated clips (FakeAVCeleb test split)",
                        value=clips[0].clip_id,
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary",
                                            size="lg")
                with gr.Row():
                    video_out = gr.Video(label="Clip", autoplay=True)
                    plot_out = gr.Image(label="Sync-score curve",
                                        type="pil")
                banner_out = gr.HTML(label="Verdict")
                explanation_out = gr.Markdown()

                analyze_btn.click(
                    on_analyze,
                    inputs=[clip_picker],
                    outputs=[video_out, banner_out, explanation_out, plot_out],
                )

            if LIVE_ENABLED:
                with gr.Tab("Live Challenge"):
                    gr.Markdown(
                        "Coming in Phase 2 — webcam + audio-swap challenge."
                    )

    return app


if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
