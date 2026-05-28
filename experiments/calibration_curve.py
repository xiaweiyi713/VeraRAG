"""Generate calibration curves from VeraBench evaluation results.

Usage:
  python experiments/calibration_curve.py --demo
  python experiments/calibration_curve.py --input results/verabench_demo.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_calibration(predicted_confidences, actual_correctness, n_bins=10):
    """Compute calibration data for plotting."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (predicted_confidences >= lo) & (predicted_confidences < hi)
        if i == n_bins - 1:
            mask |= predicted_confidences == hi

        n = mask.sum()
        if n > 0:
            avg_conf = predicted_confidences[mask].mean()
            avg_acc = actual_correctness[mask].mean()
        else:
            avg_conf = (lo + hi) / 2
            avg_acc = 0.0

        bins.append({
            "bin": i + 1,
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": int(n),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
        })

    total = len(predicted_confidences)
    ece = sum(
        abs(b["avg_confidence"] - b["avg_accuracy"]) * b["count"] / max(total, 1)
        for b in bins
    )

    return bins, float(ece)


def generate_svg(bins, ece, output_path):
    """Generate calibration curve SVG."""
    w, h = 600, 400
    margin = 60
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
    svg += 'style="font-family: Inter, system-ui, sans-serif;">\n'
    svg += f'<rect width="{w}" height="{h}" fill="#0a0a0f"/>\n'
    svg += f'<text x="{w/2}" y="25" text-anchor="middle" fill="#e5e7eb" font-size="14">'
    svg += f'Calibration Curve (ECE = {ece:.3f})</text>\n'
    svg += f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" stroke="#374151" stroke-width="1"/>\n'
    svg += f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" stroke="#374151" stroke-width="1"/>\n'
    svg += f'<text x="{margin + plot_w/2}" y="{h - 10}" text-anchor="middle" fill="#9ca3af" font-size="11">预测置信度</text>\n'
    svg += f'<text x="15" y="{margin + plot_h/2}" text-anchor="middle" fill="#9ca3af" font-size="11" '
    svg += f'transform="rotate(-90, 15, {margin + plot_h/2})">实际准确率</text>\n'
    svg += f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin}" '
    svg += 'stroke="#6b7280" stroke-width="1" stroke-dasharray="4,4"/>\n'
    svg += f'<text x="{margin + plot_w - 5}" y="{margin + 15}" text-anchor="end" fill="#6b7280" font-size="9">完美校准</text>\n'

    points = []
    for b in bins:
        if b["count"] > 0:
            x = margin + b["avg_confidence"] * plot_w
            y = margin + plot_h - b["avg_accuracy"] * plot_h
            points.append((x, y, b))
            perfect_y = margin + plot_h - b["avg_confidence"] * plot_h
            color = "#6ee7b7" if abs(b["avg_confidence"] - b["avg_accuracy"]) < 0.1 else "#fbbf24"
            svg += f'<line x1="{x}" y1="{y}" x2="{x}" y2="{perfect_y}" stroke="{color}" stroke-width="2" opacity="0.5"/>\n'
            svg += f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>\n'

    if len(points) > 1:
        path = f'M {points[0][0]},{points[0][1]}'
        for px, py, _ in points[1:]:
            path += f' L {px},{py}'
        svg += f'<path d="{path}" fill="none" stroke="#a78bfa" stroke-width="2"/>\n'

    ly = h - 30
    svg += f'<circle cx="{margin}" cy="{ly}" r="3" fill="#6ee7b7"/>'
    svg += f'<text x="{margin + 8}" y="{ly + 3}" fill="#9ca3af" font-size="9">偏差 &lt; 0.1</text>\n'
    svg += f'<circle cx="{margin + 100}" cy="{ly}" r="3" fill="#fbbf24"/>'
    svg += f'<text x="{margin + 108}" y="{ly + 3}" fill="#9ca3af" font-size="9">偏差 ≥ 0.1</text>\n'
    svg += f'<text x="{margin + 210}" y="{ly + 3}" fill="#6b7280" font-size="9">虚线 = 完美校准</text>\n'
    svg += '</svg>'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to evaluation results JSON")
    parser.add_argument("--output", default="results/calibration_curve.svg")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    args = parser.parse_args()

    if args.demo:
        np.random.seed(42)
        n = 102
        predicted = np.clip(np.random.beta(5, 2, n), 0.1, 0.95)
        actual = (np.random.random(n) < predicted * 0.85).astype(float)
    elif args.input:
        with open(args.input) as f:
            results = json.load(f)
        predicted = np.array([r["confidence"] for r in results])
        actual = np.array([1.0 if r.get("correct") else 0.0 for r in results])
    else:
        print("Specify --input or --demo")
        sys.exit(1)

    bins, ece = compute_calibration(predicted, actual)
    print(f"ECE: {ece:.4f}")
    for b in bins:
        print(f"  Bin {b['bin']} ({b['range']}): conf={b['avg_confidence']:.3f} "
              f"acc={b['avg_accuracy']:.3f} n={b['count']}")

    generate_svg(bins, ece, args.output)


if __name__ == "__main__":
    main()
