#!/usr/bin/env python3
"""Generate hand-drawn sketch architecture diagram for gentle-ai-model-router."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_architecture_diagram(out_path: Path) -> None:
    with plt.xkcd(scale=0.8, randomness=1, length=100):
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)

        # Style constants
        EDGE_COLOR = "black"
        LINE_WIDTH = 1.6

        def draw_box(x, y, w, h, title, subtitle, bg_color, title_color="black"):
            box = patches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.15,rounding_size=0.2",
                facecolor=bg_color,
                edgecolor=EDGE_COLOR,
                linewidth=LINE_WIDTH,
                zorder=2,
            )
            ax.add_patch(box)
            ax.text(
                x + w / 2, y + h * 0.65, title,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=title_color, zorder=3,
            )
            ax.text(
                x + w / 2, y + h * 0.32, subtitle,
                ha="center", va="center", fontsize=8.5,
                color="#333333", zorder=3,
            )

        def draw_arrow(x1, y1, x2, y2, label=""):
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.5",
                    lw=LINE_WIDTH, color=EDGE_COLOR, shrinkA=5, shrinkB=5,
                ),
                zorder=4,
            )
            if label:
                ax.text(
                    (x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#555555", zorder=5,
                )

        # 1. Collectors
        draw_box(
            0.8, 4.4, 3.2, 1.35,
            "1. Multi-Source Collectors",
            "Artificial Analysis, LMArena,\nSWE-Traces, RouterBench, RouteLLM",
            "#EBF5FB", "#1B4F72"
        )

        # 2. Registry
        draw_box(
            0.8, 2.4, 3.2, 1.35,
            "2. Normalized Registry",
            "Postgres / SQLite DB\nSnapshots, Provenance & Pricing",
            "#FEF9E7", "#7D6608"
        )

        # 3. Dataset Builder
        draw_box(
            4.4, 2.4, 3.2, 1.35,
            "3. Dataset Builder",
            "Temporal anti-leakage splits,\n$O(K)$ preference pairs, Parquet",
            "#EAFAF1", "#196F3D"
        )

        # 4. ModernBERT Ranker + System One
        draw_box(
            8.0, 2.4, 3.2, 1.35,
            "4. ModernBERT Ranker",
            "Hybrid 22D numeric + text encoder,\nChoice, Score, Noul multi-task heads",
            "#FDEDEC", "#922B21"
        )

        # 5. FastAPI /route
        draw_box(
            8.0, 0.4, 3.2, 1.3,
            "5. FastAPI /route Service",
            "Single-pass neural inference (19.7ms),\nCalibrated confidence, Bandit loop",
            "#F4ECF7", "#5B2C6F"
        )

        # 6. Gentle AI Adapters & Hooks
        draw_box(
            2.3, 0.4, 5.0, 1.3,
            "6. Gentle AI Integration & Hooks",
            "OpenCode, Pi, Codex, Claude adapters,\nRuntime hooks & Telemetry shim",
            "#F2F4F4", "#2C3E50"
        )

        # Connective Arrows
        draw_arrow(2.4, 4.4, 2.4, 3.75, "snapshots")
        draw_arrow(4.0, 3.07, 4.4, 3.07, "priors")
        draw_arrow(7.6, 3.07, 8.0, 3.07, "train pairs")
        draw_arrow(9.6, 2.4, 9.6, 1.7, "model.quant.onnx")
        draw_arrow(8.0, 1.05, 7.3, 1.05, "route decision")
        draw_arrow(2.3, 1.05, 1.6, 2.4, "telemetry feedback")

        # Diagram Title
        ax.text(
            6.0, 6.6, "Gentle AI Model Router — Architecture & Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#111111"
        )
        ax.text(
            6.0, 6.15, "Phase-aware minimum sufficient effort selection for SDD phases",
            ha="center", va="center", fontsize=10, style="italic",
            color="#555555"
        )

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated architecture diagram: {out_path}")

if __name__ == "__main__":
    generate_architecture_diagram(Path("docs/assets/architecture-target.png"))
