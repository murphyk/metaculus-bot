"""backtest_eval.py — evaluate ForecastReport objects against resolved outcomes."""
from __future__ import annotations

import logging
from typing import Sequence

from forecasting_tools import (
    BinaryReport,
    DateReport,
    DiscreteReport,
    ForecastReport,
    MultipleChoiceReport,
    NumericReport,
)
from forecasting_tools.data_models.questions import CanceledResolution

logger = logging.getLogger(__name__)


def generate_backtest_html(
    reports: Sequence[ForecastReport | BaseException],
    output_path: str = "backtest_results.html",
) -> None:
    """Build an HTML table comparing bot predictions to resolved outcomes."""
    valid = [r for r in reports if isinstance(r, ForecastReport)]
    rows = [_row_data(r) for r in valid]

    binary_scores = [r["brier"] for r in rows if r["brier"] is not None]
    mean_brier = sum(binary_scores) / len(binary_scores) if binary_scores else None

    html = _build_html(rows, mean_brier)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Backtest results saved to {output_path}")
    if mean_brier is not None:
        logger.info(
            f"Mean binary Brier score: {mean_brier:.4f} "
            f"(lower is better; 0 = perfect, 0.25 = random guessing)"
        )


def _row_data(report: ForecastReport) -> dict:
    q = report.question
    resolution = q.resolution_string or "—"

    try:
        pred_str = report.make_readable_prediction(report.prediction)
    except Exception:
        pred_str = str(report.prediction)

    brier = _brier_score(report)

    return {
        "url": q.page_url,
        "text": q.question_text,
        "q_type": type(q).__name__.replace("Question", ""),
        "prediction": pred_str,
        "resolution": resolution,
        "brier": brier,
    }


def _brier_score(report: ForecastReport) -> float | None:
    """Compute Brier score for resolved binary questions; None otherwise."""
    if not isinstance(report, BinaryReport):
        return None
    try:
        res = report.question.binary_resolution
        if isinstance(res, bool):
            r = 1.0 if res else 0.0
            return (report.prediction - r) ** 2
    except Exception:
        pass
    return None


def _brier_color(score: float) -> str:
    if score <= 0.1:
        return "#c6efce"  # green — good
    if score <= 0.25:
        return "#ffeb9c"  # yellow — ok
    return "#ffc7ce"      # red — poor


def _build_html(rows: list[dict], mean_brier: float | None) -> str:
    summary_html = ""
    if mean_brier is not None:
        summary_html = (
            f'<p class="summary">Binary questions: '
            f"<strong>mean Brier score = {mean_brier:.4f}</strong> "
            f"(lower is better; 0 = perfect, 0.25 = random guessing)</p>\n"
        )

    rows_html = ""
    for i, row in enumerate(rows, 1):
        if row["brier"] is not None:
            color = _brier_color(row["brier"])
            brier_cell = f'<td style="background:{color}">{row["brier"]:.4f}</td>'
        else:
            brier_cell = "<td>—</td>"
        pred_display = row["prediction"].replace("\n", "<br>")
        rows_html += (
            f"<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{row["url"]}" target="_blank">{row["text"]}</a></td>'
            f'<td class="q-type">{row["q_type"]}</td>'
            f"<td>{pred_display}</td>"
            f"<td>{row['resolution']}</td>"
            f"{brier_cell}"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Results — Fall 2025</title>
<style>
  body {{ font-family: sans-serif; padding: 1em 2em; color: #222; }}
  h1 {{ font-size: 1.4em; margin-bottom: 0.3em; }}
  .summary {{ margin-bottom: 1em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
  th {{ background: #4472c4; color: #fff; padding: 8px 10px; text-align: left; white-space: nowrap; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  a {{ color: #1a73e8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .q-type {{ font-size: 0.8em; color: #666; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Backtest Results — Fall 2025 Tournament</h1>
{summary_html}<table>
<thead><tr>
  <th>#</th><th>Question</th><th>Type</th><th>Prediction</th><th>Resolution</th><th>Brier</th>
</tr></thead>
<tbody>
{rows_html}</tbody>
</table>
</body>
</html>"""
