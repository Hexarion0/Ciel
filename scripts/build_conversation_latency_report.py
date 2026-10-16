#!/usr/bin/env python3
"""Build a private, offline A/B report from latency benchmark requests."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("data/diagnostics/conversation-latency")
    )
    args = parser.parse_args()
    path = args.directory / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["label"], row["model"]), []).append(row)
    tables = []
    for (label, model), group in sorted(groups.items()):
        cases = []
        for case in sorted({row["case"] for row in group}):
            items = [row for row in group if row["case"] == case]
            good = [row for row in items if "first_content" in row.get("timings", {})]
            timing = [row["timings"]["first_content"] for row in good]
            errors = ", ".join(
                sorted({
                    row.get("error", "HTTP error")
                    for row in items if row.get("status") != 200
                })
            )
            summary = (
                f"{statistics.median(timing):.2f}s median, "
                f"{min(timing):.2f}–{max(timing):.2f}s range"
                if timing else "No usable text"
            )
            cases.append(
                "<tr><td>" + html.escape(case) + "</td><td>" + summary + "</td><td>"
                + html.escape(errors or f"{len(good)}/{len(items)} usable") + "</td></tr>"
            )
        tables.append(
            "<section><h2>" + html.escape(model) + "</h2><p>Run: "
            + html.escape(label) + "</p><table><tr><th>Case</th><th>First content</th>"
            "<th>Result</th></tr>" + "".join(cases) + "</table></section>"
        )
    answers = []
    for row in rows:
        if not row.get("content"):
            continue
        times = row.get("timings", {})
        answers.append(
            "<article><h3>" + html.escape(row["case"]) + " · "
            + html.escape(row["model"]) + " · repeat " + str(row["repeat"] + 1)
            + "</h3><p>First usable text: "
            + html.escape(f"{times.get('first_content', float('nan')):.2f}s")
            + "</p><pre>" + html.escape(row["content"]) + "</pre></article>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIEL conversation latency A/B</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}
section,article{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:.45rem;border-bottom:1px solid #ddd}
pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem;border-radius:5px}</style>
<h1>CIEL conversation latency A/B</h1>
<p>Private local report from serial NIM streaming requests. Prompts were frozen and
identical within each case. First content is measured at the SSE consumer, not when
CIEL starts playback. Responses are included for human quality review.</p>
<h2>Timing by run</h2>""" + "".join(tables) + "<h2>Generated responses</h2>" + "".join(answers)
    (args.directory / "index.html").write_text(page, encoding="utf-8")
    print(args.directory / "index.html")


if __name__ == "__main__":
    main()
