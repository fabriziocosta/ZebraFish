from __future__ import annotations

import argparse

from src.experiment_runners import DEFAULT_12T_CONFIG_PATH, run_12t_pretraining, update_agent_run_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 12T commutative transformer pretraining experiment.")
    parser.add_argument("--config", default=str(DEFAULT_12T_CONFIG_PATH), help="YAML config path.")
    args = parser.parse_args()
    try:
        run_dir = run_12t_pretraining(args.config)
    except Exception as exc:
        update_agent_run_status(status="failed", experiment="12T", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        print(f"12T run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
