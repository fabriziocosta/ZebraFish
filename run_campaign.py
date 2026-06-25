#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.agent_campaign_loop import main as campaign_main


CAMPAIGNS: dict[str, dict[str, str]] = {
    "cnn": {
        "config": "configs/experiment_campaigns/cnn_campaign.yaml",
        "description": "10C CNN pretraining -> 13C CNN fine-tuning",
    },
    "transformer": {
        "config": "configs/experiment_campaigns/transformer_campaign.yaml",
        "description": "12T transformer pretraining -> 15T transformer fine-tuning",
    },
}


def available_campaigns_text() -> str:
    lines = ["available campaigns:"]
    for name, metadata in sorted(CAMPAIGNS.items()):
        lines.append(f"  {name:<12} {metadata['description']}  ({metadata['config']})")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a ZebraFish experiment campaign from the repo root.",
        epilog=available_campaigns_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign", help="Campaign name: cnn, transformer, or a path to a campaign YAML.")
    parser.add_argument("--poll-seconds", type=int, default=None, help="Override the campaign poll interval.")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without launching jobs, calling OpenAI, or writing files.")
    parser.add_argument("--start-trial", default=None, help="Optional explicit trial id for a new campaign trial.")
    parser.add_argument("--new-trial", action="store_true", help="Start a new trial even when campaign state already exists.")
    parser.add_argument(
        "--terminate-child-on-exit",
        action="store_true",
        help="On Ctrl-C, also terminate a child process launched by this campaign.",
    )
    return parser


def resolve_campaign(value: str) -> str:
    if value in CAMPAIGNS:
        return CAMPAIGNS[value]["config"]
    path = Path(value)
    if path.exists():
        return str(path)
    known = ", ".join(sorted(CAMPAIGNS))
    raise SystemExit(f"Unknown campaign {value!r}; use one of {known}, or pass a YAML path.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    if argv[0] == "list":
        print(available_campaigns_text())
        return 0
    if argv[0] == "status":
        if len(argv) < 2:
            print("usage: ./run_campaign status <campaign>", file=sys.stderr)
            print(available_campaigns_text(), file=sys.stderr)
            return 2
        return campaign_main(["status", "--campaign", resolve_campaign(argv[1])])
    args = parser.parse_args(argv)
    forwarded = ["run", "--campaign", resolve_campaign(args.campaign)]
    if args.poll_seconds is not None:
        forwarded.extend(["--poll-seconds", str(args.poll_seconds)])
    if args.once:
        forwarded.append("--once")
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.start_trial:
        forwarded.extend(["--start-trial", args.start_trial])
    if args.new_trial:
        forwarded.append("--new-trial")
    if args.terminate_child_on_exit:
        forwarded.append("--terminate-child-on-exit")
    return campaign_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
