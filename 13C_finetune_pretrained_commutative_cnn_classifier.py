from __future__ import annotations

import argparse

from src.experiment_runners import DEFAULT_13C_CONFIG_PATH, run_13c_finetune, update_agent_run_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 13C pretrained commutative CNN fine-tune experiment.")
    parser.add_argument("--config", default=str(DEFAULT_13C_CONFIG_PATH), help="YAML config path.")
    args = parser.parse_args()
    try:
        run_dir = run_13c_finetune(args.config)
    except Exception as exc:
        update_agent_run_status(status="failed", experiment="13C", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        print(f"13C run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
