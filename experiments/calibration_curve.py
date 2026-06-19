"""Generate calibration curves from VeraBench evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_calibration(
    predicted_confidences: np.ndarray,
    actual_correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[dict[str, Any]], float]:
    """Compute calibration data for plotting."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if len(predicted_confidences) != len(actual_correctness):
        raise ValueError("predicted_confidences and actual_correctness must have the same length")
    if len(predicted_confidences) == 0:
        raise ValueError("calibration requires at least one result row")

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins: list[dict[str, Any]] = []

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

        bins.append(
            {
                "bin": i + 1,
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": int(n),
                "avg_confidence": float(avg_conf),
                "avg_accuracy": float(avg_acc),
                "gap": float(abs(avg_conf - avg_acc)),
            }
        )

    total = len(predicted_confidences)
    ece = sum(
        abs(b["avg_confidence"] - b["avg_accuracy"]) * b["count"] / max(total, 1)
        for b in bins
    )

    return bins, float(ece)


def compute_brier_score(
    predicted_confidences: np.ndarray,
    actual_correctness: np.ndarray,
) -> float:
    """Compute mean squared calibration error."""
    if len(predicted_confidences) != len(actual_correctness):
        raise ValueError("predicted_confidences and actual_correctness must have the same length")
    if len(predicted_confidences) == 0:
        raise ValueError("brier score requires at least one result row")
    return float(np.mean((predicted_confidences - actual_correctness) ** 2))


def load_confidence_rows(
    path: str | Path,
    correctness_field: str = "correct",
) -> tuple[np.ndarray, np.ndarray]:
    """Load confidences and correctness from current or legacy result JSON."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    rows = _extract_result_rows(data)
    predicted = np.array(
        [_coerce_confidence(row.get("confidence"), idx) for idx, row in enumerate(rows)],
        dtype=float,
    )
    actual = np.array(
        [_coerce_correctness(row, correctness_field, idx) for idx, row in enumerate(rows)],
        dtype=float,
    )
    return predicted, actual


def _extract_result_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("question_results") or data.get("results") or []
    else:
        raise ValueError("calibration input must be a JSON object or list")

    if not rows:
        raise ValueError("calibration input contains no result rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("calibration result rows must be JSON objects")
    return rows


def _coerce_confidence(value: Any, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"row {row_index} confidence must be a number in [0, 1]")
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"row {row_index} confidence must be a finite value in [0, 1]")
    return confidence


def _coerce_correctness(
    row: dict[str, Any],
    correctness_field: str,
    row_index: int,
) -> float:
    if correctness_field not in row:
        raise ValueError(f"row {row_index} missing correctness field {correctness_field!r}")
    value = row[correctness_field]
    if not isinstance(value, bool):
        raise ValueError(f"row {row_index} correctness field {correctness_field!r} must be boolean")
    return 1.0 if value else 0.0


def generate_svg(
    bins: list[dict[str, Any]],
    ece: float,
    output_path: str | Path,
) -> None:
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
    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write(svg)


def _summary(
    bins: list[dict[str, Any]],
    ece: float,
    brier_score: float,
    *,
    rows: int,
    correctness_field: str,
    output: str,
) -> dict[str, Any]:
    return {
        "rows": rows,
        "correctness_field": correctness_field,
        "expected_calibration_error": round(ece, 6),
        "brier_score": round(brier_score, 6),
        "output": output,
        "bins": bins,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to evaluation results JSON")
    source.add_argument("--demo", action="store_true", help="Use deterministic demo data")
    parser.add_argument("--output", default="results/calibration_curve.svg")
    parser.add_argument(
        "--correctness-field",
        default="correct",
        help="Boolean result-row field to use as the calibration outcome.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    parser.add_argument("--json-output", help="Optional path for the machine-readable summary.")
    args = parser.parse_args(argv)

    if args.demo:
        np.random.seed(42)
        n = 102
        predicted = np.clip(np.random.beta(5, 2, n), 0.1, 0.95)
        actual = (np.random.random(n) < predicted * 0.85).astype(float)
    else:
        try:
            predicted, actual = load_confidence_rows(args.input, args.correctness_field)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.exit(2, f"calibration input error: {exc}\n")

    try:
        bins, ece = compute_calibration(predicted, actual, n_bins=args.bins)
        brier_score = compute_brier_score(predicted, actual)
    except ValueError as exc:
        parser.exit(2, f"calibration error: {exc}\n")

    generate_svg(bins, ece, args.output)
    summary = _summary(
        bins,
        ece,
        brier_score,
        rows=len(predicted),
        correctness_field=args.correctness_field,
        output=args.output,
    )
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"ECE: {ece:.4f}")
        print(f"Brier: {brier_score:.4f}")
        for b in bins:
            print(
                f"  Bin {b['bin']} ({b['range']}): conf={b['avg_confidence']:.3f} "
                f"acc={b['avg_accuracy']:.3f} n={b['count']}"
            )
        print(f"Saved: {args.output}")
        if args.json_output:
            print(f"Saved summary: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
