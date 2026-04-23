#!/usr/bin/env python
"""SyncGuard research-showcase demo (Gradio).

Tab 1: curated gallery — click a clip to analyze.
Tab 2: live challenge (only if SYNCGUARD_LIVE=1) — webcam + audio swap.

Runs entirely on the Mac. No HPC, no internet required at demo time.
"""
import os
import sys
import tempfile
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


def _bar(label, prob, color):
    pct = int(round(prob * 100))
    return f"""
    <div style="margin:8px 0 !important;">
        <div style="display:flex !important; justify-content:space-between !important;
                    font-size:15px !important; color:#ffffff !important;
                    font-weight:600 !important; margin-bottom:5px !important;">
            <span style="color:#ffffff !important;">{label}</span>
            <span style="color:#ffffff !important;">{pct}% fake</span>
        </div>
        <div style="background:#4a5568 !important; border-radius:4px !important;
                    height:14px !important; overflow:hidden !important;
                    border:1px solid #2d3748 !important;">
            <div style="background:{color} !important; width:{pct}% !important;
                        height:100% !important;"></div>
        </div>
    </div>
    """


def verdict_banner_html(verdict, confidence, ground_truth,
                        sync_prob=None, audio_prob=None,
                        mean_sync=None, num_dips=None):
    color = C_FAKE if verdict == "fake" else C_REAL
    icon = "FAKE" if verdict == "fake" else "REAL"
    correct = "Correct" if verdict == ground_truth else "Incorrect"
    badge_color = C_REAL if verdict == ground_truth else C_FAKE

    signals_html = ""
    if sync_prob is not None and audio_prob is not None:
        curve_stats = ""
        if mean_sync is not None:
            curve_stats = f"""
            <div style="margin-top:12px !important; padding-top:10px !important;
                        border-top:1px solid #4a5568 !important;
                        font-size:15px !important; color:#ffffff !important;">
                <strong style="color:#ffffff !important;">Sync curve:</strong>
                <span style="color:#ffffff !important;">mean =
                <span style="font-weight:700 !important; color:#ffffff !important;">{mean_sync:.3f}</span>,
                dips below {SYNC_THRESHOLD:.2f} =
                <span style="font-weight:700 !important; color:#ffffff !important;">{num_dips}</span>
                </span>
            </div>
            """
        signals_html = f"""
        <div style="padding:14px 18px !important; border-radius:10px !important;
                    background:#1a202c !important; border:2px solid #2d3748 !important;
                    margin-top:14px !important; font-family:system-ui !important;
                    color:#ffffff !important;">
            <div style="font-size:13px !important; color:#a0aec0 !important;
                        font-weight:700 !important; text-transform:uppercase !important;
                        margin-bottom:10px !important; letter-spacing:1px !important;">
                Per-head signals (cascade inputs)
            </div>
            {_bar("Sync head (v4+CA)", sync_prob, "#4299e1")}
            {_bar("Audio classifier", audio_prob, "#b794f4")}
            {curve_stats}
        </div>
        """

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
    {signals_html}
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
            sync_prob=result.sync_prob, audio_prob=result.audio_prob,
            mean_sync=result.mean_sync,
            num_dips=len([s for s, e in result.sync_dip_segments
                          if (e - s) >= 0.15]),
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
                from src.demo.live_challenge import process_live_clip
                from src.preprocessing.face_detector import FaceDetector

                print("Initializing MediaPipe face detector for live tab...")
                live_face_detector = FaceDetector(crop_size=96,
                                                  confidence_threshold=0.5)

                def on_live_analyze(recorded_path):
                    if recorded_path is None:
                        return (None, "", "Please record a clip first.", None)
                    try:
                        proc = process_live_clip(recorded_path, live_face_detector)
                    except Exception as e:
                        return (None, "",
                                f"**Error:** {type(e).__name__}: {e}",
                                None)
                    video_for_model = recorded_path
                    plot_title = "Sync score — your recording"
                    expected = "real"

                    if proc["detection_rate"] < 0.3:
                        return (str(video_for_model), "",
                                f"**Face detection failed** on "
                                f"{proc['detection_rate']*100:.0f}% of frames. "
                                f"Try again facing the camera with good lighting.",
                                None)

                    # Live tab: skip the cascade audio classifier — it's
                    # calibrated on FakeAVCeleb audio and mis-fires on
                    # webcam voices (every live clip would be "fake").
                    # Verdict uses the sync head alone, which measures
                    # audio-visual alignment geometrically and generalizes.
                    result = infer.analyze(
                        mouth_crops=proc["mouth_crops"],
                        audio_waveform=proc["audio_waveform"],
                        ear_features=proc["ear_features"],
                        clip_duration_s=proc["duration_s"],
                        use_audio_head=False,
                    )
                    plot = render_sync_plot(
                        result.sync_curve, SYNC_THRESHOLD,
                        result.sync_dip_segments, infer.fps_sync,
                        title=plot_title,
                    )
                    # Live tab: omit per-head signals panel. The audio
                    # classifier is out-of-distribution on webcam mic
                    # audio and would show misleading ~99% fake even on
                    # genuine real recordings. The sync-only verdict is
                    # already what the banner reflects.
                    banner = verdict_banner_html(
                        result.verdict, result.confidence,
                        ground_truth=expected,
                    )
                    explanation = generate_explanation(result)
                    return str(video_for_model), banner, explanation, plot

                with gr.Tab("Live Challenge"):
                    gr.Markdown(
                        "### Test the Model on Your Own Recording\n"
                        "Record a 3-5 second clip of yourself speaking. "
                        "The model should classify it as **REAL** — "
                        "confirming it doesn't false-positive on genuine "
                        "webcam recordings.\n\n"
                        "For the model's fake-detection capability on actual "
                        "deepfake content (wav2lip, face-swap, etc.), see "
                        "the **Gallery** tab — 0.96 AUC on the FakeAVCeleb "
                        "test split."
                    )
                    with gr.Row():
                        webcam_rec = gr.Video(
                            sources=["webcam"],
                            include_audio=True,
                            label="Record yourself (press record, speak, stop)",
                        )
                        live_analyze_btn = gr.Button(
                            "Analyze", variant="primary", size="lg",
                        )
                    with gr.Row():
                        swapped_video_out = gr.Video(
                            label="Your recording", autoplay=True,
                        )
                        live_plot_out = gr.Image(
                            label="Sync-score curve", type="pil",
                        )
                    live_banner_out = gr.HTML()
                    live_explanation_out = gr.Markdown()

                    live_analyze_btn.click(
                        on_live_analyze,
                        inputs=[webcam_rec],
                        outputs=[
                            swapped_video_out,
                            live_banner_out,
                            live_explanation_out,
                            live_plot_out,
                        ],
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
