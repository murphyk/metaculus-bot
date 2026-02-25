"""backtest_pipeline.py — multi-bot forecasting evaluation pipeline.

Directory layout produced:
  results/{tournament}/truth.json             — question metadata + resolutions
  results/{tournament}/{bot_name}/preds.json  — one bot's predictions
  results/{tournament}/combined.json          — truth merged with all bots' predictions
  configs/{bot_name}_cfg.json                 — bot config snapshot

The Metaculus REST API does not expose resolution values for regular users, so
'resolution' in truth.json is null by default.  Populate it manually to enable
Brier score computation:
  binary          → true / false
  multiple_choice → the winning option_name string (e.g. "Yes")
  numeric/discrete→ the resolved float value

Run standalone to migrate an existing raw JSON report file:
  python backtest_pipeline.py path/to/Forecasts-*.json \\
      --bot-name spring_template_2026 --tournament fall_2025
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from forecasting_tools import ForecastReport, MetaculusQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _nearest_percentile_value(declared_percentiles: list[dict], target: float) -> float:
    return min(declared_percentiles, key=lambda p: abs(p["percentile"] - target))["value"]


def _fmt_number(value: float) -> str:
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.2f}"


def _prediction_to_readable(prediction, question_type: str) -> str:
    if prediction is None:
        return "—"
    if question_type == "binary":
        return f"{float(prediction):.0%}"
    if question_type in ("numeric", "discrete"):
        if isinstance(prediction, dict) and "declared_percentiles" in prediction:
            dp = prediction["declared_percentiles"]
            p10 = _nearest_percentile_value(dp, 0.1)
            p50 = _nearest_percentile_value(dp, 0.5)
            p90 = _nearest_percentile_value(dp, 0.9)
            return f"p10={_fmt_number(p10)}, p50={_fmt_number(p50)}, p90={_fmt_number(p90)}"
        return str(prediction)
    if question_type == "multiple_choice":
        if isinstance(prediction, dict) and "predicted_options" in prediction:
            opts = sorted(
                prediction["predicted_options"], key=lambda x: -x["probability"]
            )
            return ", ".join(
                f"{o['option_name']}: {o['probability']:.0%}" for o in opts
            )
        return str(prediction)
    return str(prediction)


def _serialize_prediction(prediction) -> object:
    if prediction is None:
        return None
    if isinstance(prediction, (int, float)):
        return float(prediction)
    try:
        return json.loads(prediction.model_dump_json())
    except AttributeError:
        pass
    try:
        return json.loads(json.dumps(prediction, default=str))
    except Exception:
        return str(prediction)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def _compute_brier(prediction, question_type: str, resolution) -> float | None:
    if resolution is None or prediction is None:
        return None
    try:
        if question_type == "binary":
            r = 1.0 if resolution in (True, 1, "true", "True", "yes", "Yes") else 0.0
            return round((float(prediction) - r) ** 2, 6)
        if question_type == "multiple_choice":
            if isinstance(prediction, dict) and "predicted_options" in prediction:
                opts = {
                    str(o["option_name"]): float(o["probability"])
                    for o in prediction["predicted_options"]
                }
                total = sum(
                    (prob - (1.0 if name == str(resolution) else 0.0)) ** 2
                    for name, prob in opts.items()
                )
                return round(total, 6)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# save_truth
# ---------------------------------------------------------------------------

def save_truth(
    questions: Sequence[MetaculusQuestion],
    tournament_name: str,
    results_dir: str = "results",
) -> Path:
    """Save question metadata to results/{tournament}/truth.json.

    Previously entered resolution values are preserved if truth.json already
    exists so that manual edits survive re-runs.
    """
    out_dir = Path(results_dir) / tournament_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "truth.json"

    existing_resolutions: dict[str, object] = {}
    if out_path.exists():
        try:
            existing_resolutions = {
                qid: q["resolution"]
                for qid, q in json.loads(out_path.read_text())["questions"].items()
            }
        except Exception:
            pass

    questions_dict: dict[str, dict] = {}
    for q in questions:
        qid = str(q.id_of_post)
        questions_dict[qid] = {
            "id": q.id_of_post,
            "url": q.page_url,
            "text": q.question_text,
            "type": getattr(q, "question_type", None),
            "options": getattr(q, "options", None),
            "resolution": existing_resolutions.get(qid, None),
            "resolution_time": (
                q.actual_resolution_time.isoformat()
                if getattr(q, "actual_resolution_time", None)
                else None
            ),
            "state": getattr(q, "state", None),
        }

    payload = {
        "tournament": tournament_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "resolution is null because the Metaculus API does not expose resolution "
            "values for regular users. Manually set each question's 'resolution' field: "
            "binary → true/false, multiple_choice → winning option_name string, "
            "numeric/discrete → resolved float value."
        ),
        "questions": questions_dict,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Saved truth to {out_path} ({len(questions_dict)} questions)")
    return out_path


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------

def save_predictions(
    reports: Sequence[ForecastReport | BaseException],
    bot_name: str,
    tournament_name: str,
    results_dir: str = "results",
) -> Path:
    """Save bot predictions to results/{tournament}/{bot_name}/preds.json."""
    out_dir = Path(results_dir) / tournament_name / bot_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preds.json"

    preds: dict[str, dict] = {}
    for r in reports:
        if not isinstance(r, ForecastReport):
            continue
        q = r.question
        qid = str(q.id_of_post)
        qt = getattr(q, "question_type", "unknown") or "unknown"
        raw = _serialize_prediction(r.prediction)
        preds[qid] = {
            "prediction": raw,
            "readable": _prediction_to_readable(raw, qt),
        }

    payload = {
        "bot_name": bot_name,
        "tournament": tournament_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Saved predictions to {out_path} ({len(preds)} questions)")
    return out_path


# ---------------------------------------------------------------------------
# save_bot_config
# ---------------------------------------------------------------------------

def save_bot_config(bot, bot_name: str, configs_dir: str = "configs") -> Path:
    """Save bot configuration snapshot to configs/{bot_name}_cfg.json."""
    Path(configs_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(configs_dir) / f"{bot_name}_cfg.json"

    def _llm_repr(v):
        try:
            return {"model": v.model, "temperature": getattr(v, "temperature", None)}
        except AttributeError:
            return str(v)

    raw_llms = getattr(bot, "llms", None) or {}
    llms_repr = {k: _llm_repr(v) for k, v in raw_llms.items()}

    cfg = {
        "bot_name": bot_name,
        "class": type(bot).__name__,
        "research_reports_per_question": getattr(bot, "research_reports_per_question", None),
        "predictions_per_research_report": getattr(bot, "predictions_per_research_report", None),
        "use_research_summary_to_forecast": getattr(bot, "use_research_summary_to_forecast", None),
        "publish_reports_to_metaculus": getattr(bot, "publish_reports_to_metaculus", None),
        "llms": llms_repr,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(cfg, indent=2))
    logger.info(f"Saved bot config to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# migrate_reports_json
# ---------------------------------------------------------------------------

def migrate_reports_json(
    json_path: str,
    bot_name: str,
    tournament_name: str,
    results_dir: str = "results",
) -> tuple[Path, Path]:
    """Convert an existing forecasting_tools raw JSON report file to the new format.

    Returns (truth_path, preds_path).  Previously entered resolutions in an
    existing truth.json are preserved.
    """
    with open(json_path, encoding="utf-8") as f:
        raw_reports = json.load(f)

    out_dir = Path(results_dir) / tournament_name
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = out_dir / "truth.json"

    existing_resolutions: dict[str, object] = {}
    if truth_path.exists():
        try:
            existing_resolutions = {
                qid: q["resolution"]
                for qid, q in json.loads(truth_path.read_text())["questions"].items()
            }
        except Exception:
            pass

    questions_dict: dict[str, dict] = {}
    for r in raw_reports:
        q = r["question"]
        qid = str(q["id_of_post"])
        questions_dict[qid] = {
            "id": q["id_of_post"],
            "url": q["page_url"],
            "text": q["question_text"],
            "type": q.get("question_type"),
            "options": q.get("options"),
            "resolution": existing_resolutions.get(qid, q.get("resolution_string")),
            "resolution_time": q.get("actual_resolution_time"),
            "state": q.get("state"),
        }

    truth_payload = {
        "tournament": tournament_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "resolution is null because the Metaculus API does not expose resolution "
            "values for regular users. Manually set each question's 'resolution' field: "
            "binary → true/false, multiple_choice → winning option_name string, "
            "numeric/discrete → resolved float value."
        ),
        "questions": questions_dict,
    }
    truth_path.write_text(json.dumps(truth_payload, indent=2, default=str))
    logger.info(f"Saved truth to {truth_path} ({len(questions_dict)} questions)")

    preds_dir = out_dir / bot_name
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / "preds.json"

    preds: dict[str, dict] = {}
    for r in raw_reports:
        q = r["question"]
        qid = str(q["id_of_post"])
        qt = q.get("question_type") or "unknown"
        raw = r.get("prediction")
        preds[qid] = {
            "prediction": raw,
            "readable": _prediction_to_readable(raw, qt),
        }

    preds_payload = {
        "bot_name": bot_name,
        "tournament": tournament_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds,
    }
    preds_path.write_text(json.dumps(preds_payload, indent=2, default=str))
    logger.info(f"Saved predictions to {preds_path} ({len(preds)} questions)")

    return truth_path, preds_path


# ---------------------------------------------------------------------------
# merge → combined.json
# ---------------------------------------------------------------------------

def merge(tournament_name: str, results_dir: str = "results") -> Path:
    """Merge truth.json + all bots' preds.json into combined.json."""
    base = Path(results_dir) / tournament_name
    truth_path = base / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"truth.json not found at {truth_path}")

    truth_data = json.loads(truth_path.read_text())
    questions = truth_data["questions"]

    bots = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "preds.json").exists()
    )

    combined_questions: dict[str, dict] = {}
    for qid, q in questions.items():
        combined_questions[qid] = {
            "id": q["id"],
            "url": q["url"],
            "text": q["text"],
            "type": q["type"],
            "options": q.get("options"),
            "resolution": q.get("resolution"),
            "resolution_time": q.get("resolution_time"),
            "predictions": {},
        }

    for bot_name in bots:
        bot_preds = json.loads((base / bot_name / "preds.json").read_text())["predictions"]
        for qid, p in bot_preds.items():
            if qid not in combined_questions:
                continue
            entry = combined_questions[qid]
            brier = _compute_brier(p["prediction"], entry["type"], entry["resolution"])
            entry["predictions"][bot_name] = {
                "prediction": p["prediction"],
                "readable": p["readable"],
                "brier": brier,
            }

    payload = {
        "tournament": tournament_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bots": bots,
        "questions": combined_questions,
    }
    out_path = base / "combined.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Saved combined results to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# generate_html
# ---------------------------------------------------------------------------

def generate_html(
    tournament_name: str,
    results_dir: str = "results",
    output_html: str | None = None,
) -> Path:
    """Build an HTML comparison table from combined.json (auto-runs merge first)."""
    combined_path = Path(results_dir) / tournament_name / "combined.json"
    merge(tournament_name, results_dir)
    data = json.loads(combined_path.read_text())

    bots = data["bots"]
    questions = data["questions"]
    if output_html is None:
        output_html = f"{tournament_name}_backtest.html"

    # Per-bot mean Brier (binary only)
    bot_briers: dict[str, list[float]] = {b: [] for b in bots}
    for q in questions.values():
        if q["type"] != "binary":
            continue
        for bot_name, pred_info in q.get("predictions", {}).items():
            if pred_info.get("brier") is not None:
                bot_briers[bot_name].append(pred_info["brier"])
    mean_briers = {
        b: (sum(v) / len(v) if v else None)
        for b, v in bot_briers.items()
    }

    missing_resolution = sum(1 for q in questions.values() if q.get("resolution") is None)
    has_any_resolution = missing_resolution < len(questions)

    html = _build_html(questions, bots, mean_briers, tournament_name, missing_resolution, has_any_resolution)
    out_path = Path(output_html)
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Backtest HTML saved to {out_path}")
    return out_path


def _brier_color(score: float) -> str:
    if score <= 0.1:
        return "#c6efce"
    if score <= 0.25:
        return "#ffeb9c"
    return "#ffc7ce"


def _build_html(
    questions: dict,
    bots: list[str],
    mean_briers: dict[str, float | None],
    tournament_name: str,
    missing_resolution: int,
    has_any_resolution: bool,
) -> str:
    # --- summary section ---
    summary_parts: list[str] = []
    if missing_resolution > 0:
        summary_parts.append(
            f'<p class="warn">⚠ {missing_resolution} of {len(questions)} questions have no resolution value. '
            f'Edit <code>results/{tournament_name}/truth.json</code> to populate the '
            f'<code>"resolution"</code> field for each question, then re-run '
            f'<code>python backtest_pipeline.py --regenerate-html {tournament_name}</code>.</p>'
        )
    if has_any_resolution and bots:
        rows_html_summary = "".join(
            f"<tr><td>{b}</td>"
            f"<td>{'N/A' if mean_briers.get(b) is None else f'{mean_briers[b]:.4f}'}</td>"
            f"<td>({len(bot_briers_list(questions, b))} binary Qs)</td></tr>\n"
            for b in bots
        )
        summary_parts.append(
            '<table class="summary-table"><thead><tr>'
            '<th>Bot</th><th>Mean Binary Brier ↓</th><th>Coverage</th>'
            '</tr></thead><tbody>\n'
            + rows_html_summary
            + "</tbody></table>\n"
            '<p class="note">Brier score: lower is better. 0 = perfect, 0.25 = random guessing.</p>'
        )
    summary_html = "\n".join(summary_parts)

    # --- bot prediction header columns ---
    bot_headers = "".join(
        f"<th>Prediction<br><small>{b}</small></th>"
        f"<th>Brier<br><small>{b}</small></th>"
        for b in bots
    )

    # --- table rows ---
    rows_html = ""
    for i, (qid, q) in enumerate(questions.items(), 1):
        res = q.get("resolution")
        res_display = str(res) if res is not None else '<span class="null">—</span>'
        pred_cols = ""
        for bot_name in bots:
            pred_info = q.get("predictions", {}).get(bot_name)
            if pred_info is None:
                pred_cols += '<td class="null">—</td><td class="null">—</td>'
            else:
                readable = pred_info["readable"].replace("\n", "<br>")
                brier = pred_info.get("brier")
                if brier is not None:
                    color = _brier_color(brier)
                    brier_cell = f'<td style="background:{color}">{brier:.4f}</td>'
                else:
                    brier_cell = '<td class="null">—</td>'
                pred_cols += f"<td>{readable}</td>{brier_cell}"

        q_type = (q.get("type") or "").replace("_", " ")
        rows_html += (
            f"<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{q["url"]}" target="_blank">{q["text"]}</a></td>'
            f'<td class="q-type">{q_type}</td>'
            f"<td>{res_display}</td>"
            f"{pred_cols}"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Results — {tournament_name.replace("_", " ").title()}</title>
<style>
  body {{ font-family: sans-serif; padding: 1em 2em; color: #222; }}
  h1 {{ font-size: 1.4em; margin-bottom: 0.5em; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffc107; padding: 0.5em 1em;
           border-radius: 4px; margin-bottom: 1em; }}
  .note {{ color: #666; font-size: 0.85em; margin: 0.3em 0 1em; }}
  .summary-table {{ border-collapse: collapse; margin-bottom: 1em; }}
  .summary-table th, .summary-table td {{ border: 1px solid #ccc; padding: 5px 12px; }}
  .summary-table th {{ background: #4472c4; color: #fff; }}
  table.main {{ border-collapse: collapse; width: 100%; font-size: 0.87em; }}
  table.main th {{ background: #4472c4; color: #fff; padding: 8px 10px;
                   text-align: left; white-space: nowrap; }}
  table.main td {{ padding: 5px 10px; border-bottom: 1px solid #ddd;
                   vertical-align: top; }}
  table.main tr:nth-child(even) td {{ background: #f7f7f7; }}
  a {{ color: #1a73e8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .q-type {{ font-size: 0.8em; color: #666; white-space: nowrap; }}
  .null {{ color: #aaa; }}
</style>
</head>
<body>
<h1>Backtest Results — {tournament_name.replace("_", " ").title()}</h1>
{summary_html}
<table class="main">
<thead><tr>
  <th>#</th><th>Question</th><th>Type</th><th>Resolution</th>{bot_headers}
</tr></thead>
<tbody>
{rows_html}</tbody>
</table>
</body>
</html>"""


def bot_briers_list(questions: dict, bot_name: str) -> list[float]:
    """Return list of non-null Brier scores for bot across binary questions."""
    result = []
    for q in questions.values():
        if q["type"] != "binary":
            continue
        pred_info = q.get("predictions", {}).get(bot_name)
        if pred_info and pred_info.get("brier") is not None:
            result.append(pred_info["brier"])
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backtest pipeline utilities")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # migrate subcommand
    m = subparsers.add_parser("migrate", help="Migrate raw JSON reports to new format")
    m.add_argument("json_path", help="Path to existing Forecasts-*.json file")
    m.add_argument("--bot-name", default="spring_template_2026")
    m.add_argument("--tournament", default="fall_2025")
    m.add_argument("--results-dir", default="results")

    # html subcommand
    h = subparsers.add_parser("html", help="Regenerate HTML from existing combined.json")
    h.add_argument("tournament", help="Tournament name (e.g. fall_2025)")
    h.add_argument("--results-dir", default="results")
    h.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.cmd == "migrate":
        migrate_reports_json(args.json_path, args.bot_name, args.tournament, args.results_dir)
        merge(args.tournament, args.results_dir)
        generate_html(args.tournament, args.results_dir)
        print(f"Done. HTML saved to {args.tournament}_backtest.html")
    elif args.cmd == "html":
        out = generate_html(args.tournament, args.results_dir, args.output)
        print(f"Done. HTML saved to {out}")
