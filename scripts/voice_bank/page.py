"""Build a private local listening page; raw and prepared audio remain reviewable."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .audio import timestamp


def audio(directory: Path, relative: str | None, label: str, extra: str = "") -> str:
    """Render valid audio links or a clear pending/failed state."""
    safe_label = html.escape(label)
    if not relative or not (directory / relative).is_file():
        return (
            f'<div class="clip"><h4>{safe_label}</h4>'
            '<p class="muted">Not generated yet.</p></div>'
        )
    return (
        f'<div class="clip"><h4>{safe_label}</h4><audio controls preload="none" '
        f'aria-label="{safe_label}" src="{html.escape(relative, quote=True)}" {extra}></audio>'
        '<p class="audio-error" hidden>Audio unavailable. Keep the page beside its audio files.</p>'
        '</div>'
    )


def render(directory: Path) -> Path:
    """Extend the project's restrained report layout with listening/feedback controls."""
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    keys = {}
    for row in manifest["candidates"]:
        identifier = row["id"]
        safe_id = html.escape(identifier, quote=True)
        baseline = row["source_id"] == "baseline"
        style = row["intended_style"]
        provenance = html.escape(row.get("source_timestamp", ""))
        if row.get("source_url"):
            url = row["source_url"] + "&t=" + str(int(row.get("start", 0)))
            provenance = (
                f'<a href="{html.escape(url, quote=True)}" '
                f'target="_blank" rel="noopener">{html.escape(row["source_id"])}</a> · '
                + provenance
            )
        heading = "Existing Ciel baseline" if baseline else identifier
        blind = row.get("blind", {})
        keys[identifier] = {label: value["mode"].upper() for label, value in blind.items()}
        ab = "".join(audio(
            directory, blind.get(label, {}).get("audio"), f"Pocket sample {label}",
            f'data-candidate="{safe_id}" data-blind="{label}"',
        ) for label in ("A", "B"))
        original = audio(directory, row.get("unprocessed"), "Unprocessed source clip")
        reference = audio(directory, row.get("reference"), "Prepared reference")
        style_output = row.get("previews", {}).get("fp32", {}).get("style", {})
        common_text = html.escape(
            manifest.get("preview_configuration", {}).get("common_text", ""),
        )
        style_text = html.escape(style_output.get("text", ""))
        style_audio = audio(directory, style_output.get("audio"), "Pocket style-specific line")
        options = "".join(
            f'<option value="{item}" {"selected" if item == style else ""}>{item}</option>'
            for item in ("neutral", "warm", "amused", "firm", "focused", "soft", "unclear")
        )
        m = row.get("metrics", {})
        info = "Preserved baseline; compare against every new reference."
        if m:
            cosine = row.get("speaker_cosine_to_anchor")
            proxy = f"{cosine:.3f}" if cosine is not None else "pending recheck"
            info = (
                f"Original {row['original_duration_seconds']:.2f}s → "
                f"prepared {row['processed_duration_seconds']:.2f}s. "
                f"Linear gain {row['preprocessing']['gain_db']:+.1f} dB; "
                "no denoising or separation. "
                f"Speaker proxy {proxy}; "
                "delivery remains unconfirmed."
            )
        rows.append(f'''
<section class="candidate" data-style="{style}" id="{safe_id}">
  <div class="candidate-heading"><h2>{html.escape(heading)}</h2>
  <span>{html.escape(style)} · {"baseline" if baseline else "provisional"}</span></div>
  <p class="provenance">{provenance}</p><p class="muted">{html.escape(info)}</p>
  <div class="audio-pair">{original}{reference}</div>
  <h3>Same sentence · blind numerical-mode comparison</h3>
  <p class="test-line">{common_text}</p>
  <div class="audio-pair">{ab}</div>
  <p><button class="reveal" data-candidate="{safe_id}" disabled>Reveal A/B after listening</button>
  <span class="reveal-result" id="reveal-{safe_id}" aria-live="polite"></span></p>
  <details><summary>Style-specific test</summary>
    <p class="test-line">{style_text}</p>{style_audio}</details>
  <div class="feedback" data-candidate="{safe_id}">
    <label>Reference decision<select data-field="decision">
    <option value="review">Needs review</option><option value="keep">Keep on shortlist</option>
    <option value="reject">Reject</option></select></label>
    <label>Delivery you hear<select data-field="style">{options}</select></label>
    <label>Numerical-mode preference<select data-field="preference">
    <option value="unrated">Not rated</option><option>A</option><option>B</option>
    <option value="equal">No meaningful difference</option></select></label>
    <label class="notes">Listening notes<textarea rows="2" data-field="notes"
    placeholder="Similarity, diction, artifacts, warmth, pacing, naturalness, emotion"></textarea>
    </label>
  </div>
</section>''')
    source_rows = []
    for source in manifest["sources"]:
        source_rows.append(
            '<tr><td>' + html.escape(source["id"]) + '</td><td>' + html.escape(source["title"])
            + '</td><td>' + timestamp(source["duration_seconds"]) + '</td><td>'
            + html.escape(source["review_status"]) + '</td></tr>'
        )
    settings = {
        "run_id": directory.name, "blind_key": keys,
        "candidate_ids": [row["id"] for row in manifest["candidates"]],
    }
    template = Path(__file__).with_name("audition.html").read_text()
    page = template.replace("__CANDIDATES__", "".join(rows)).replace(
        "__SOURCE_ROWS__", "".join(source_rows),
    ).replace("__RUN__", html.escape(directory.name)).replace(
        "__SETTINGS__", json.dumps(settings).replace("<", "\\u003c"),
    )
    path = directory / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
