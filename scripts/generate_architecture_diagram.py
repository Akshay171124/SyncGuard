"""Generate SyncGuard architecture diagram for presentation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
C_INPUT = '#E8F4FD'       # Light blue - inputs
C_PREPROCESS = '#D5E8D4'  # Light green - preprocessing
C_ENCODER = '#DAE8FC'     # Blue - encoders
C_PRETRAINED = '#B8D4E3'  # Darker blue - pretrained
C_LOSS = '#FFF2CC'        # Yellow - losses
C_CLASSIFIER = '#E1D5E7'  # Purple - classifier
C_OUTPUT = '#F8CECC'      # Red/pink - output
C_ARROW = '#333333'
C_TEXT = '#1A1A1A'
C_BORDER = '#666666'

def draw_box(x, y, w, h, text, color, fontsize=9, bold=False, border_color=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=border_color or C_BORDER,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + w / 2, y + h / 2, text,
        ha='center', va='center',
        fontsize=fontsize, fontweight=weight,
        color=C_TEXT, zorder=3,
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # center_bottom, center_top

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle='arc3,rad=0',
        ),
        zorder=1,
    )

def draw_curved_arrow(x1, y1, x2, y2, color=C_ARROW, rad=0.3):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=1.5,
            connectionstyle=f'arc3,rad={rad}',
        ),
        zorder=1,
    )

# Title
ax.text(8, 8.6, 'SyncGuard: Contrastive Audio-Visual Deepfake Detection',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1A5276')
ax.text(8, 8.2, 'System Architecture',
        ha='center', va='center', fontsize=12, fontweight='normal', color='#666666')

# ===== INPUT LAYER (left) =====
draw_box(0.3, 5.8, 2.0, 1.0, 'Video\nInput', C_INPUT, fontsize=10, bold=True)
draw_box(0.3, 3.2, 2.0, 1.0, 'Audio\nInput', C_INPUT, fontsize=10, bold=True)

# ===== PREPROCESSING =====
draw_box(3.0, 5.8, 2.4, 1.0, 'RetinaFace +\nMediaPipe\nMouth ROI\n(96×96)', C_PREPROCESS, fontsize=8)
draw_box(3.0, 3.2, 2.4, 1.0, 'FFmpeg +\nSilero-VAD\n16kHz Mono', C_PREPROCESS, fontsize=8)

# Arrows: input -> preprocess
draw_arrow(2.3, 6.3, 3.0, 6.3)
draw_arrow(2.3, 3.7, 3.0, 3.7)

# ===== ENCODERS =====
draw_box(6.1, 5.5, 2.6, 1.6, 'AV-HuBERT\nVisual Encoder\n\n3D Conv + ResNet-18\n→ Projection Head\n→ L2 Normalize', C_ENCODER, fontsize=8, bold=False, border_color='#3498DB')
draw_box(6.1, 2.9, 2.6, 1.6, 'Wav2Vec 2.0\nAudio Encoder\n\nLayer 9 Hidden States\n→ Projection Head\n→ L2 Normalize', C_ENCODER, fontsize=8, bold=False, border_color='#3498DB')

# Arrows: preprocess -> encoders
draw_arrow(5.4, 6.3, 6.1, 6.3)
draw_arrow(5.4, 3.7, 6.1, 3.7)

# Dimension labels
ax.text(9.0, 6.0, 'v_t ∈ ℝ^{B×T×256}', fontsize=8, fontstyle='italic', color='#2E86C1')
ax.text(9.0, 3.4, 'a_t ∈ ℝ^{B×T×256}', fontsize=8, fontstyle='italic', color='#2E86C1')

# ===== SYNC SCORE =====
draw_box(9.8, 4.5, 2.2, 1.2, 'Cosine\nSimilarity\n\ns(t) = cos(v_t, a_t)', C_LOSS, fontsize=8, bold=False, border_color='#F39C12')

# Arrows: encoders -> sync score
draw_arrow(8.7, 5.8, 9.8, 5.3)
draw_arrow(8.7, 4.2, 9.8, 4.9)

# ===== CLASSIFIER =====
draw_box(12.6, 4.5, 2.2, 1.2, 'Bi-LSTM\nClassifier\n\n2-layer, hidden=128\nMean+Max Pool', C_CLASSIFIER, fontsize=8, bold=False, border_color='#8E44AD')

# Arrow: sync score -> classifier
draw_arrow(12.0, 5.1, 12.6, 5.1)

# ===== OUTPUT =====
draw_box(12.6, 2.5, 2.2, 0.9, 'Real / Fake\nPrediction', C_OUTPUT, fontsize=10, bold=True, border_color='#E74C3C')

# Arrow: classifier -> output
draw_arrow(13.7, 4.5, 13.7, 3.4)

# ===== LOSSES (bottom) =====
# InfoNCE
draw_box(3.5, 0.5, 2.5, 1.2, 'InfoNCE Loss\n+ MoCo Queue\n(4096 negatives)', C_LOSS, fontsize=8, border_color='#F39C12')

# Temporal
draw_box(6.5, 0.5, 2.5, 1.2, 'Temporal\nConsistency Loss\n(real clips only)', C_LOSS, fontsize=8, border_color='#F39C12')

# BCE
draw_box(9.5, 0.5, 2.0, 1.2, 'BCE Loss\n(classification)', C_LOSS, fontsize=8, border_color='#F39C12')

# Combined
draw_box(12.0, 0.5, 2.8, 1.2, 'L = L_nce +\nγ·L_temp + δ·L_cls\n(γ=0.5, δ=1.0)', C_LOSS, fontsize=8, bold=True, border_color='#E67E22')

# Arrows from encoders to losses
draw_curved_arrow(7.4, 2.9, 4.75, 1.7, color='#999999', rad=0.2)
draw_curved_arrow(7.4, 2.9, 7.75, 1.7, color='#999999', rad=0.1)

# Arrows from classifier output to BCE
draw_curved_arrow(13.7, 2.5, 10.5, 1.7, color='#999999', rad=-0.3)

# Arrows losses -> combined
draw_arrow(6.0, 1.1, 12.0, 1.1)
draw_arrow(9.0, 1.1, 12.0, 1.1)
draw_arrow(11.5, 1.1, 12.0, 1.1)

# ===== TRAINING PHASES (annotation) =====
# Phase 1 box
ax.add_patch(FancyBboxPatch(
    (3.3, 0.3), 2.9, 1.6,
    boxstyle="round,pad=0.15", facecolor='none',
    edgecolor='#27AE60', linewidth=2, linestyle='--', zorder=1,
))
ax.text(4.75, 2.05, 'Phase 1', fontsize=7, color='#27AE60', ha='center', fontweight='bold')

# Phase 2 box
ax.add_patch(FancyBboxPatch(
    (3.3, 0.3), 11.7, 1.6,
    boxstyle="round,pad=0.2", facecolor='none',
    edgecolor='#E74C3C', linewidth=2, linestyle=':', zorder=0,
))
ax.text(14.5, 2.05, 'Phase 2', fontsize=7, color='#E74C3C', ha='center', fontweight='bold')

# Legend
legend_y = 7.6
ax.add_patch(FancyBboxPatch((0.3, legend_y - 0.1), 0.3, 0.3, boxstyle="round,pad=0.02",
             facecolor=C_PREPROCESS, edgecolor=C_BORDER, linewidth=1))
ax.text(0.8, legend_y + 0.05, 'Preprocessing', fontsize=7, va='center')

ax.add_patch(FancyBboxPatch((2.5, legend_y - 0.1), 0.3, 0.3, boxstyle="round,pad=0.02",
             facecolor=C_ENCODER, edgecolor='#3498DB', linewidth=1))
ax.text(3.0, legend_y + 0.05, 'Encoders (pretrained)', fontsize=7, va='center')

ax.add_patch(FancyBboxPatch((5.2, legend_y - 0.1), 0.3, 0.3, boxstyle="round,pad=0.02",
             facecolor=C_CLASSIFIER, edgecolor='#8E44AD', linewidth=1))
ax.text(5.7, legend_y + 0.05, 'Classifier', fontsize=7, va='center')

ax.add_patch(FancyBboxPatch((7.2, legend_y - 0.1), 0.3, 0.3, boxstyle="round,pad=0.02",
             facecolor=C_LOSS, edgecolor='#F39C12', linewidth=1))
ax.text(7.7, legend_y + 0.05, 'Loss Functions', fontsize=7, va='center')

# Frozen label
ax.text(6.4, 7.15, '❄ Wav2Vec 2.0 backbone frozen', fontsize=7, color='#2980B9', fontstyle='italic')

plt.tight_layout()
plt.savefig('outputs/visualizations/architecture_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('outputs/visualizations/architecture_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Architecture diagram saved to outputs/visualizations/')
