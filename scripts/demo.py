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
    <div style="margin:4px 0;">
        <div style="display:flex; justify-content:space-between;
                    font-size:13px; color:#333; margin-bottom:2px;">
            <span>{label}</span><span>{pct}% fake</span>
        </div>
        <div style="background:#e0e0e0; border-radius:4px; height:10px;
                    overflow:hidden;">
            <div style="background:{color}; width:{pct}%; height:100%;"></div>
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
            <div style="margin-top:8px; font-size:13px; color:#444;">
                <strong>Sync curve:</strong>
                mean = {mean_sync:.3f},
                dips below {SYNC_THRESHOLD:.2f} = {num_dips}
            </div>
            """
        signals_html = f"""
        <div style="padding:10px 16px; border-radius:8px; background:#fafafa;
                    margin-top:10px; font-family:system-ui;">
            <div style="font-size:12px; color:#666; font-weight:600;
                        text-transform:uppercase; margin-bottom:6px;">
                Per-head signals (cascade inputs)
            </div>
            {_bar("Sync head (v4+CA)", sync_prob, "#1A5276")}
            {_bar("Audio classifier", audio_prob, "#8E44AD")}
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
                from src.demo.live_challenge import swap_audio, process_live_clip
                from src.preprocessing.face_detector import FaceDetector

                print("Initializing MediaPipe face detector for live tab...")
                live_face_detector = FaceDetector(crop_size=96,
                                                  confidence_threshold=0.5)

                def on_live_analyze(recorded_path, mode):
                    """mode: 'as-is' (no swap, expect real) or 'swap' (expect fake)."""
                    if recorded_path is None:
                        return (None, "", "Please record a clip first.", None)
                    tmpdir = Path(tempfile.mkdtemp(prefix="syncguard_live_"))
                    try:
                        if mode == "swap":
                            video_for_model = tmpdir / "swapped.mp4"
                            chosen = swap_audio(recorded_path, video_for_model)
                            expected = "fake"
                            plot_title = f"Sync score — swapped audio: {chosen}"
                        else:
                            video_for_model = recorded_path
                            expected = "real"
                            plot_title = "Sync score — your recording (unaltered)"
                        proc = process_live_clip(video_for_model, live_face_detector)
                    except Exception as e:
                        return (None, "",
                                f"**Error:** {type(e).__name__}: {e}",
                                None)

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
                    banner = verdict_banner_html(
                        result.verdict, result.confidence,
                        ground_truth=expected,
                        sync_prob=result.sync_prob,
                        audio_prob=result.audio_prob,
                        mean_sync=result.mean_sync,
                        num_dips=len([s for s, e in result.sync_dip_segments
                                     if (e - s) >= 0.15]),
                    )
                    explanation = generate_explanation(result)
                    return str(video_for_model), banner, explanation, plot

                with gr.Tab("Live Challenge"):
                    gr.Markdown(
                        "### Record Yourself and Test the Model\n"
                        "Record a 3-5 second clip of yourself speaking. "
                        "Choose a mode:\n"
                        "- **Analyze as-is:** keep your real audio. The model "
                        "should classify you as **REAL**.\n"
                        "- **Swap audio (lip-sync challenge):** we replace "
                        "your audio with a random sentence from our pool, "
                        "creating a lip-sync mismatch. The model should "
                        "detect it as **FAKE**."
                    )
                    with gr.Row():
                        webcam_rec = gr.Video(
                            sources=["webcam"],
                            include_audio=True,
                            label="Record yourself (press record, speak, stop)",
                        )
                        with gr.Column():
                            mode_picker = gr.Radio(
                                choices=[
                                    ("Analyze as-is (expect REAL)", "as-is"),
                                    ("Swap audio (expect FAKE)", "swap"),
                                ],
                                label="Mode",
                                value="as-is",
                            )
                            live_analyze_btn = gr.Button(
                                "Analyze",
                                variant="primary", size="lg",
                            )
                    with gr.Row():
                        swapped_video_out = gr.Video(
                            label="What the model sees", autoplay=True,
                        )
                        live_plot_out = gr.Image(
                            label="Sync-score curve", type="pil",
                        )
                    live_banner_out = gr.HTML()
                    live_explanation_out = gr.Markdown()

                    live_analyze_btn.click(
                        on_live_analyze,
                        inputs=[webcam_rec, mode_picker],
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
