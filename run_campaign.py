#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from src.scientific_state import load_state


def load_campaign_config(path):
    try:
        from src.agent_campaign_loop import load_campaign_config as _load_campaign_config

        return _load_campaign_config(path)
    except ModuleNotFoundError as exc:
        if exc.name not in {"pandas", "torch"}:
            raise
        target = Path(path)
        text = target.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
        campaign = payload.setdefault("campaign", {})
        campaign.setdefault("id", target.stem)
        campaign.setdefault("scientific_state_path", "state/scientific_state.yaml")
        payload.setdefault("scientific_state", {"path": campaign["scientific_state_path"]})
        payload.setdefault("artifacts", {})
        payload["artifacts"].setdefault("root", f"artifacts/campaigns/{campaign['id']}")
        payload["artifacts"].setdefault("state_path", f"{payload['artifacts']['root']}/campaign_state.json")
        return payload


def campaign_live_status(config):
    from src.agent_campaign_loop import campaign_live_status as _campaign_live_status

    return _campaign_live_status(config)


def campaign_main(argv):
    from src.agent_campaign_loop import main as _campaign_main

    return _campaign_main(argv)


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


def command_help_text() -> str:
    return "\n".join(
        [
            "commands:",
            "  ./run_campaign <campaign> [options]    start or resume a campaign loop",
            "  ./run_campaign status <campaign>       print campaign status without launching",
            "  ./run_campaign suspend [campaign]      request cooperative suspension after the current epoch",
            "  ./run_campaign resume [campaign]       resume a suspended campaign stage",
            "  ./run_campaign terminate [campaign]    terminate the active training child",
            "  ./run_campaign force-restart <campaign> clean up and start a fresh campaign trial",
            "  ./run_campaign migrate-state <campaign> import historical campaign evidence",
            "  ./run_campaign state [campaign]             print the scientific state",
            "  ./run_campaign observations [campaign]      print deterministic observations",
            "  ./run_campaign candidates [campaign]        print autonomous candidates",
            "  ./run_campaign meta-controller <campaign>  run the daily supervisory controller",
            "  ./run_campaign rebuild-views <campaign>     rebuild compatibility views",
            "  ./run_campaign dashboard [campaign]         serve the read-only mission-control dashboard",
            "  ./run_campaign list                    list available campaigns",
            "",
            available_campaigns_text(),
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="run_campaign.py <campaign>|status <campaign>|terminate [campaign]|force-restart <campaign>|list [options]",
        description="Start a ZebraFish experiment campaign from the repo root.",
        epilog=command_help_text(),
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
    for metadata in CAMPAIGNS.values():
        try:
            if load_campaign_config(metadata["config"])["campaign"]["id"] == value:
                return metadata["config"]
        except Exception:
            continue
    path = Path(value)
    if path.exists():
        return str(path)
    known = ", ".join(sorted(CAMPAIGNS))
    raise SystemExit(f"Unknown campaign {value!r}; use one of {known}, or pass a YAML path.")


def _known_live_campaigns() -> list[tuple[str, dict[str, str], dict[str, object]]]:
    live = []
    for name, metadata in CAMPAIGNS.items():
        try:
            config = load_campaign_config(metadata["config"])
            status = campaign_live_status(config)
        except Exception:
            continue
        if status.get("running"):
            live.append((name, metadata, status))
    return live


def resolve_default_live_campaign() -> tuple[str | None, int]:
    live = _known_live_campaigns()
    if not live:
        return None, 0
    if len(live) > 1:
        print("multiple running campaigns found; specify one explicitly:")
        for name, metadata, status in sorted(live, key=lambda item: item[0]):
            print(f"  {name:<12} campaign_id={status.get('campaign_id')} pid={status.get('pid')} config={metadata['config']}")
        return None, 2
    live.sort(key=lambda item: float(item[2].get("state_mtime") or 0), reverse=True)
    name, metadata, status = live[0]
    print(
        f"selected live campaign {name} ({status.get('campaign_id')}) "
        f"pid={status.get('pid')} from {metadata['config']}"
    )
    return metadata["config"], 0


def terminate_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Terminate the active training child for a campaign.",
        epilog=available_campaigns_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign", nargs="?", help="Campaign name, campaign id, or YAML path. Defaults only when exactly one known campaign is live.")
    parser.add_argument("--campaign-id", default=None, help="Campaign id such as cnn_pretrain_finetune.")
    parser.add_argument("--reason", default="terminated by run_campaign CLI", help="Reason recorded in campaign state.")
    parser.add_argument("--force-after", type=float, default=None, help="Seconds after SIGTERM before SIGKILL escalation.")
    parser.add_argument("--require-running", action="store_true", help="Return nonzero if no running process is found.")
    args = parser.parse_args(argv)
    if args.force_after is not None and args.force_after <= 0:
        parser.error("--force-after must be greater than 0")
    target = args.campaign_id or args.campaign
    if target:
        config_path = resolve_campaign(target)
    else:
        config_path, code = resolve_default_live_campaign()
        if config_path is None:
            if code == 0:
                print("no running campaign found")
                return 1 if args.require_running else 0
            return code
    forwarded = ["terminate", "--campaign", config_path, "--reason", args.reason]
    if args.force_after is not None:
        forwarded.extend(["--force-after", str(args.force_after)])
    if args.require_running:
        forwarded.append("--require-running")
    return campaign_main(forwarded)


def suspend_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Request cooperative suspension after the active epoch finishes.",
        epilog=available_campaigns_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign", nargs="?", help="Campaign name, campaign id, or YAML path. Defaults only when exactly one known campaign is live.")
    parser.add_argument("--campaign-id", default=None, help="Campaign id such as cnn_pretrain_finetune.")
    parser.add_argument("--reason", default="suspend requested by run_campaign CLI", help="Reason written to the suspend marker.")
    args = parser.parse_args(argv)
    target = args.campaign_id or args.campaign
    if target:
        config_path = resolve_campaign(target)
    else:
        config_path, code = resolve_default_live_campaign()
        if config_path is None:
            if code == 0:
                print("no running campaign found")
                return 1
            return code
    return campaign_main(["suspend", "--campaign", config_path, "--reason", args.reason])


def resume_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Resume a campaign stage suspended at an epoch boundary.",
        epilog=available_campaigns_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign", nargs="?", help="Campaign name, campaign id, or YAML path. Defaults only when exactly one known campaign is live.")
    parser.add_argument("--campaign-id", default=None, help="Campaign id such as cnn_pretrain_finetune.")
    args = parser.parse_args(argv)
    target = args.campaign_id or args.campaign
    if target:
        config_path = resolve_campaign(target)
    else:
        config_path, code = resolve_default_live_campaign()
        if config_path is None:
            if code == 0:
                print("no running campaign found")
                return 1
            return code
    return campaign_main(["resume", "--campaign", config_path])


def force_restart_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Terminate any active/stale processes for a campaign and start a fresh trial.",
        epilog=available_campaigns_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign", help="Campaign name, campaign id, or YAML path.")
    parser.add_argument("--reason", default="force-restart requested by run_campaign CLI", help="Reason recorded in campaign state.")
    parser.add_argument("--force-after", type=float, default=5.0, help="Seconds after SIGTERM before SIGKILL escalation. Use 0 to escalate immediately.")
    parser.add_argument("--start-trial", default=None, help="Optional explicit trial id for the fresh campaign trial.")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle after restart and exit.")
    args = parser.parse_args(argv)
    if args.force_after is not None and args.force_after < 0:
        parser.error("--force-after must be greater than or equal to 0")
    forwarded = [
        "force-restart",
        "--campaign",
        resolve_campaign(args.campaign),
        "--reason",
        args.reason,
        "--force-after",
        str(args.force_after),
    ]
    if args.start_trial:
        forwarded.extend(["--start-trial", args.start_trial])
    if args.once:
        forwarded.append("--once")
    return campaign_main(forwarded)


def dashboard_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Serve the local read-only mission-control dashboard.")
    parser.add_argument("campaign", nargs="?", default="cnn", help="Campaign alias or id (default: cnn).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; defaults to localhost only.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--reload", action="store_true", help="Reload the API during development.")
    args = parser.parse_args(argv)
    # Validate before starting the server so a typo does not produce a blank UI.
    resolve_campaign(args.campaign)
    os.environ["ZEBRAFISH_DASHBOARD_CAMPAIGN"] = args.campaign
    # Generate concise semantic graph labels once per unchanged node while the
    # dashboard is running.  The API remains read-only; labels are in-memory
    # presentation metadata and failures fall back to deterministic labels.
    os.environ.setdefault("ZEBRAFISH_DASHBOARD_LLM_LABELS", "on")
    import uvicorn

    uvicorn.run("src.dashboard_api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def meta_controller_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run or control the daily supervisory meta-controller.")
    parser.add_argument("campaign", help="Campaign alias, campaign id, or YAML path.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one bounded diagnosis/remediation pass.")
    mode.add_argument("--start", action="store_true", help="Run the persistent daily loop.")
    mode.add_argument("--continue", dest="continue_loop", action="store_true", help="Continue the persistent daily loop.")
    mode.add_argument("--stop", action="store_true", help="Stop the persistent daily loop.")
    args = parser.parse_args(argv)
    from src.meta_controller import cli as meta_cli

    forwarded = [resolve_campaign(args.campaign)]
    forwarded.append("--once" if args.once else "--start" if args.start else "--continue" if args.continue_loop else "--stop")
    return meta_cli(forwarded, root=Path(__file__).resolve().parent)


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
    if argv[0] == "terminate":
        return terminate_command(argv[1:])
    if argv[0] == "suspend":
        return suspend_command(argv[1:])
    if argv[0] == "resume":
        return resume_command(argv[1:])
    if argv[0] == "force-restart":
        return force_restart_command(argv[1:])
    if argv[0] == "dashboard":
        return dashboard_command(argv[1:])
    if argv[0] == "meta-controller":
        return meta_controller_command(argv[1:])
    if argv[0] in {"state", "observations", "candidates"}:
        target = resolve_campaign(argv[1]) if len(argv) > 1 else CAMPAIGNS["cnn"]["config"]
        config = load_campaign_config(target)
        state = load_state(config.get("scientific_state", {}).get("path", "state/scientific_state.yaml"))
        if argv[0] == "observations":
            payload = state.get("entities", {}).get("observations", {})
        elif argv[0] == "candidates":
            payload = state.get("entities", {}).get("candidate_experiments", {})
        else:
            payload = state
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if argv[0] == "migrate-state":
        if len(argv) < 2:
            print("usage: ./run_campaign migrate-state <campaign>", file=sys.stderr)
            return 2
        config = load_campaign_config(resolve_campaign(argv[1]))
        from src.state_migration import migrate_campaign

        result = migrate_campaign(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if argv[0] == "rebuild-views":
        if len(argv) < 2:
            print("usage: ./run_campaign rebuild-views <campaign>", file=sys.stderr)
            return 2
        config = load_campaign_config(resolve_campaign(argv[1]))
        from src.state_migration import rebuild_compatibility_views

        print(json.dumps(rebuild_compatibility_views(config), indent=2, sort_keys=True))
        return 0
    args = parser.parse_args(argv)
    config = load_campaign_config(resolve_campaign(args.campaign))
    if args.poll_seconds is not None:
        config["campaign"]["poll_seconds"] = args.poll_seconds
    from src.autonomous_campaign import run_autonomous_campaign

    return run_autonomous_campaign(
        config,
        once=args.once,
        dry_run=args.dry_run,
        start_trial_id=args.start_trial,
        new_trial=args.new_trial,
        terminate_child_on_exit=args.terminate_child_on_exit,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
