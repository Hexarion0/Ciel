"""Build an immutable, local audition run for permission-confirmed Ciel references."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from scripts.voice_bank.audio import SOURCE_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    downloader = sub.add_parser("download", help="Archive confirmed sources without overwriting")
    downloader.add_argument("--source-root", type=Path, default=Path("data/voice/source/ciel"))
    analyzer = sub.add_parser("analyze", help="Screen whole sources and retain raw comparisons")
    analyzer.add_argument("--source-root", type=Path, default=Path("data/voice/source/ciel"))
    analyzer.add_argument("--output", type=Path, required=True)
    selector = sub.add_parser(
        "shortlist", help="Use CPU speaker-screening results; still provisional",
    )
    selector.add_argument("directory", type=Path)
    selector.add_argument("--baseline", type=Path,
                          default=Path("data/voice/references/ciel/reference-1.wav"))
    preview = sub.add_parser("preview", help="Export voice states; generate CPU FP32/INT8 samples")
    preview.add_argument("directory", type=Path)
    preview.add_argument("--cache", type=Path, default=Path("data/voice/states/ciel"))
    preview.add_argument("--language", default="english_2026-09")
    page = sub.add_parser("page", help="Render the static local listening page")
    page.add_argument("directory", type=Path)
    stress = sub.add_parser("stress", help="Paced CPU synthesis/caching test; does not play audio")
    stress.add_argument("directory", type=Path)
    stress.add_argument("--candidate", default="candidate-neutral-01")
    stress.add_argument("--seconds", type=float, default=300)
    verifier = sub.add_parser(
        "verify", help="CPU ASR and edge checks; human listening still needed",
    )
    verifier.add_argument("directory", type=Path)
    extractor = sub.add_parser("extract", help="Add a new precisely chosen source clip")
    extractor.add_argument("directory", type=Path)
    extractor.add_argument("--source-root", type=Path, default=Path("data/voice/source/ciel"))
    extractor.add_argument("--id", required=True)
    extractor.add_argument("--source", choices=SOURCE_IDS, required=True)
    extractor.add_argument("--start", type=float, required=True)
    extractor.add_argument("--end", type=float, required=True)
    extractor.add_argument("--style", choices=["neutral", "warm", "amused", "firm",
                                              "focused", "soft"], required=True)
    extractor.add_argument("--notes", default="")
    reporter = sub.add_parser("report", help="Write the unaccepted proposal and findings")
    reporter.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        from scripts.voice_bank.analyze import permission

        permission(args.source_root)
        executable = shutil.which("yt-dlp")
        if importlib.util.find_spec("yt_dlp"):
            launcher = [sys.executable, "-m", "yt_dlp"]
        elif executable:
            launcher = [executable]
        else:
            raise RuntimeError('Install declared dependencies: pip install -e ".[voice-bank]"')
        pending = [sid for sid in SOURCE_IDS if not all(
            (args.source_root / f"source-{sid}{suffix}").is_file()
            for suffix in (".webm", ".info.json", ".en-orig.vtt")
        )]
        if not pending:
            print("All four source archives are complete; nothing downloaded or overwritten.")
            return
        command = [*launcher, "--js-runtimes", "node", "--no-playlist", "--no-overwrites",
                   "--continue", "--write-info-json", "--write-description", "--no-progress",
                   "--retries", "2", "--socket-timeout", "20", "-f", "bestaudio",
                   "-o", str(args.source_root / "source-%(id)s.%(ext)s")]
        # One original-language caption track avoids translated-track rate limits.
        command += ["--write-auto-subs", "--sub-langs", "en-orig", "--sub-format", "vtt"]
        command += [f"https://www.youtube.com/watch?v={sid}" for sid in pending]
        subprocess.run(command, check=True)
        (args.source_root / "download-command.json").write_text(json.dumps(command, indent=2))
    elif args.command == "analyze":
        from scripts.voice_bank.analyze import run

        print(run(args.source_root, args.output))
    elif args.command == "shortlist":
        from scripts.voice_bank.select import shortlist

        print(shortlist(args.directory, args.baseline))
    elif args.command == "preview":
        from scripts.voice_bank.preview import run

        run(args.directory, args.cache, args.language)
    elif args.command == "page":
        from scripts.voice_bank.page import render

        print(render(args.directory))
    elif args.command == "stress":
        from scripts.voice_bank.stress import run

        if args.seconds < 1:
            parser.error("Stress duration must be positive")
        print(run(args.directory, args.candidate, args.seconds))
    elif args.command == "verify":
        from scripts.voice_bank.verify import run

        print(run(args.directory))
    elif args.command == "extract":
        from scripts.voice_bank.extract import run

        print(run(args.directory, args.source_root, identifier=args.id, source_id=args.source,
                  start=args.start, end=args.end, style=args.style, notes=args.notes))
    elif args.command == "report":
        from scripts.voice_bank.report import run

        print(run(args.directory))


if __name__ == "__main__":
    main()
