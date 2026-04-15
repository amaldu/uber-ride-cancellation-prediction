"""CLI entry point: python -m analysis <command>

Commands:
    run     Run analysis pipeline, export to SQLite, generate dashboard, start Grafana
    analyze Run analysis pipeline only (no Grafana)
    serve   Start Grafana (assumes analysis has been run)
    stop    Stop Grafana
"""

import argparse
import logging
import sys

from analysis.run import analyze, serve, stop


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m analysis",
        description="Uber Ride Cancellation Analysis Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run analysis + start Grafana dashboard")
    p_run.add_argument("--csv", help="Path to raw CSV (default: data/raw/ncr_ride_bookings.csv)")

    p_analyze = sub.add_parser("analyze", help="Run analysis only (no Grafana)")
    p_analyze.add_argument("--csv", help="Path to raw CSV")

    sub.add_parser("serve", help="Start Grafana (analysis must have been run)")
    sub.add_parser("stop", help="Stop Grafana")

    args = parser.parse_args()

    if args.command == "run":
        analyze(raw_csv=args.csv)
        serve()
    elif args.command == "analyze":
        analyze(raw_csv=args.csv)
    elif args.command == "serve":
        serve()
    elif args.command == "stop":
        stop()


if __name__ == "__main__":
    main()
