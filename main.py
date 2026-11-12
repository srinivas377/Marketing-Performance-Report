#!/usr/bin/env python3
"""Command-line entry point.

  python main.py                 run once and write output/report.html + report.json
  python main.py --watch         run, then re-run automatically whenever a CSV changes
  python main.py --no-llm        skip the API call (rule-based actions only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from analyzer import __version__
from analyzer.config import PROJECT_ROOT, AppConfig
from analyzer.pipeline import run_once
from analyzer.utils import DataError
from analyzer.watcher import watch


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Marketing Performance Analyzer")
    p.add_argument("--seo", type=Path, default=PROJECT_ROOT / "data" / "seo_data.csv", help="SEO CSV path")
    p.add_argument("--ads", type=Path, default=PROJECT_ROOT / "data" / "ads_data.csv", help="Ads CSV path")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "output", help="Output folder")
    p.add_argument("--provider", choices=["anthropic", "openai", "gemini", "grok"], help="LLM provider (default: env LLM_PROVIDER or anthropic)")
    p.add_argument("--model", help="Model name (default: env LLM_MODEL or the provider default)")
    p.add_argument("--no-llm", action="store_true", help="Skip the LLM call")
    p.add_argument("--watch", action="store_true", help="Re-run automatically when a CSV file changes")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between file checks in --watch mode")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    cfg = AppConfig(seo_path=args.seo, ads_path=args.ads, output_dir=args.out,
                    use_llm=not args.no_llm, provider=args.provider, model=args.model)
    try:
        run_once(cfg)
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if not args.watch:
            return 2
    if args.watch:
        print(f"\nWatching {cfg.seo_path.name} and {cfg.ads_path.name} (Ctrl+C to stop) ...")

        def refresh():
            print("\nData changed – regenerating report ...")
            run_once(cfg)

        try:
            watch([cfg.seo_path, cfg.ads_path], refresh, interval=args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
