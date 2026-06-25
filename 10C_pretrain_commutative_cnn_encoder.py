from __future__ import annotations

import argparse

from src.experiment_runners import DEFAULT_10C_CONFIG_PATH, run_10c_pretraining, update_agent_run_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 10C commutative CNN pretraining experiment.")
    parser.add_argument("--config", default=str(DEFAULT_10C_CONFIG_PATH), help="YAML config path.")
    args = parser.parse_args()
    try:
        run_dir = run_10c_pretraining(args.config)
    except Exception as exc:
        update_agent_run_status(status="failed", experiment="10C", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        print(f"10C run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
