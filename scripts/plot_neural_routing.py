#!/usr/bin/env python3
"""Generate hand-drawn sketch diagram illustrating ModernBERT v14 task-aware decision routing."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_neural_routing_diagram(out_path: Path) -> None:
    with plt.xkcd(scale=0.8, randomness=1, length=100):
        fig, ax = plt.subplots(figsize=(13.2, 7.6), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(0, 13.2)
        ax.set_ylim(0, 7.6)

        # Style constants
        EDGE_COLOR = "black"
        LINE_WIDTH = 1.6

        def draw_box(x, y, w, h, title, subtitle, bg_color, title_color="black"):
            box = patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.15,rounding_size=0.2",
                facecolor=bg_color,
                edgecolor=EDGE_COLOR,
                linewidth=LINE_WIDTH,
                zorder=2,
            )
            ax.add_patch(box)
            t1 = ax.text(
                x + w / 2,
                y + h * 0.76,
                title,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )
            t1.set_path_effects([])
            t2 = ax.text(
                x + w / 2,
                y + h * 0.38,
                subtitle,
                ha="center",
                va="center",
                fontsize=8.3,
                color="#222222",
                zorder=3,
            )
            t2.set_path_effects([])

        def draw_arrow(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            label: str = "",
            label_pos: tuple[float, float] | None = None,
            rotation: float = 0.0,
        ):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.5",
                    lw=LINE_WIDTH,
                    color=EDGE_COLOR,
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=4,
            )
            if label:
                lx = (x1 + x2) / 2 if label_pos is None else label_pos[0]
                ly = (y1 + y2) / 2 if label_pos is None else label_pos[1]
                t_arr = ax.text(
                    lx,
                    ly,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    fontweight="bold",
                    color="#222222",
                    rotation=rotation,
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="#FFFFFF",
                        edgecolor="#AAAAAA",
                        linewidth=0.8,
                    ),
                    zorder=6,
                )
                t_arr.set_path_effects([])

        # Column 1 (Inputs)
        task_text = (
            "• SDD Phase: explore, verify, apply...\n"
            "• Task Prompt: e.g.\n"
            "  'Run test suite, verify boundaries'\n"
            "• Context Tokens: 24,000\n"
            "• Per-Phase Quality Floor: P_min"
        )
        draw_box(
            0.6,
            4.0,
            3.4,
            2.7,
            "1. Task Request & Context",
            task_text,
            "#E8F8F5",
            "#0E6251",
        )

        candidates_text = (
            "Discovered from Registry (data/router.db):\n"
            "• kimi-for-coding @ off\n"
            "• openai/gpt-5.5 @ off\n"
            "• deepseek-v4-pro @ low\n"
            "• deepseek-v4-pro @ high\n"
            "(Only valid provider-supported variants)"
        )
        draw_box(
            0.6,
            0.6,
            3.4,
            2.8,
            "2. Candidate Arms Pool",
            candidates_text,
            "#FEF9E7",
            "#7E5109",
        )

        # Column 2 (Feature Fusion)
        cross_text = (
            "Cross-Encoder Text Format:\n"
            "[phase] sdd-verify\n"
            "[task] Run test suite, verify boundaries\n"
            "[candidate] deepseek-v4-pro effort=low\n\n"
            "(Encodes task complexity vs candidate)"
        )
        draw_box(
            4.6,
            4.0,
            3.7,
            2.7,
            "3. Joint Prompt Tokenization",
            cross_text,
            "#EBF5FB",
            "#1B4F72",
        )

        dense_text = (
            "Model Capabilities & Benchmarks:\n"
            "• tool_calling, structured_output\n"
            "• log(context_window), log(max_output)\n"
            "• Speed proxy & pricing ($/1M in/out)\n"
            "• Normalised Category Priors (Arena, AA)"
        )
        draw_box(
            4.6,
            0.6,
            3.7,
            2.8,
            "4. Dense Feature Vector (22D)",
            dense_text,
            "#EAFAF1",
            "#196F3D",
        )

        # Column 3 (Model Inference & Selection)
        model_text = (
            "Cross-Encoder ONNX INT8 (255MB)\n"
            "• Choice Head: Arm affinity logit\n"
            "• Score Head: Expected effort rubric\n"
            "• Noul Head: Fast success prob P(succ)\n"
            "Latency: 19.7ms (RTX 5070) / 59.8ms (CPU)"
        )
        draw_box(
            8.9,
            4.0,
            3.7,
            2.7,
            "5. ModernBERT Core (v14)",
            model_text,
            "#FDEDEC",
            "#922B21",
        )

        decision_text = (
            "Minimum Sufficient Effort Selection:\n"
            "• verify floor: Q >= 80%\n"
            "• off variant fails floor (Q=75%)\n"
            "• low variant meets floor (Q=85.7%)\n"
            "• WINNER: deepseek-v4-pro @ low\n"
            "Savings: -46.2% tokens vs static high"
        )
        draw_box(
            8.9,
            0.6,
            3.7,
            2.8,
            "6. Optimal Route Decision",
            decision_text,
            "#FFF9C4",
            "#6E2C00",
        )

        # Clean, non-overlapping flow arrows
        draw_arrow(4.0, 5.35, 4.6, 5.35, "Task Text", label_pos=(4.3, 5.35))
        draw_arrow(
            4.0, 2.7, 4.6, 4.5, "Arm Pairing", label_pos=(4.3, 3.6), rotation=60.0
        )
        draw_arrow(4.0, 1.8, 4.6, 1.8, "Tech Specs", label_pos=(4.3, 1.8))
        draw_arrow(8.3, 5.35, 8.9, 5.35, "Token Pairs", label_pos=(8.6, 5.35))
        draw_arrow(
            8.3, 2.7, 8.9, 4.5, "22D Dense", label_pos=(8.6, 3.6), rotation=60.0
        )
        draw_arrow(
            10.75, 4.0, 10.75, 3.4, "Rank & Gate", label_pos=(10.75, 3.7)
        )

        # Diagram Title & Subtitle
        t_tit = fig.text(
            0.5,
            0.96,
            "How ModernBERT v14 Decides (Model, Effort) per Task",
            ha="center",
            va="top",
            fontsize=13.5,
            fontweight="bold",
            color="black",
        )
        t_tit.set_path_effects([])
        subtitle_text = (
            "Joint cross-encoder text tokenization + dense feature fusion "
            "with Minimum Sufficient Effort enforcement"
        )
        t_sub = fig.text(
            0.5,
            0.92,
            subtitle_text,
            ha="center",
            va="top",
            fontsize=9.0,
            color="#555555",
        )
        t_sub.set_path_effects([])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Generated neural routing diagram: {out_path}")


if __name__ == "__main__":
    generate_neural_routing_diagram(Path("docs/assets/neural-routing-flow.png"))
