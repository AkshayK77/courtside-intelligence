"""
run_pipeline.py — Run the full Courtside Intelligence pipeline in one command.

Usage:
    python run_pipeline.py           # fetch + train + launch dashboard
    python run_pipeline.py --skip-fetch   # skip BBRef scraping, train on existing CSVs
    python run_pipeline.py --dashboard-only  # just launch the dashboard
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import importlib
import time

# ── Colour helpers (no dependencies) ─────────────────────────────────────────
def _c(code, text): return f"\033[{code}m{text}\033[0m"
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


def step(n: int, title: str):
    print(f"\n{bold(f'── Step {n}/4 — {title}')}")
    print(dim("─" * 50))


def run_step(module_name: str) -> bool:
    """Import and run module_name.main(). Returns True on success."""
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)   # re-exec if already imported
        mod.main()
        return True
    except SystemExit:
        return True             # some scripts call sys.exit(0) on success
    except Exception as exc:
        print(red(f"\n✗ {module_name} raised an error: {exc}"))
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Courtside Intelligence full pipeline")
    parser.add_argument("--skip-fetch",      action="store_true",
                        help="Skip building historical_teams.csv; use existing file")
    parser.add_argument("--skip-training",   action="store_true",
                        help="Skip model training; use existing win_model.pkl")
    parser.add_argument("--dashboard-only",  action="store_true",
                        help="Skip all data steps and just launch the dashboard")
    args = parser.parse_args()

    start = time.time()
    print(bold("\n╔══════════════════════════════════════════════╗"))
    print(bold("║   Courtside Intelligence — Full Pipeline     ║"))
    print(bold("╚══════════════════════════════════════════════╝"))

    # ── Step 1: build historical team stats from local Kaggle CSV ────────────
    if not args.dashboard_only and not args.skip_fetch:
        step(1, "Build historical team stats from stats_since_1950/")
        ok = run_step("fetch_historical")
        if ok:
            print(green("  ✓ historical_teams.csv written"))
        else:
            print(yellow("  ⚠  fetch_historical had errors"))
    else:
        print(dim("\n  [Step 1 skipped — using existing historical_teams.csv]"))

    # ── Step 2: merge historical + current season into training set ───────────
    if not args.dashboard_only and not args.skip_fetch:
        step(2, "Merge historical + current season → training_data.csv")
        ok = run_step("build_training_data")
        if ok:
            print(green("  ✓ training_data.csv written"))
    else:
        print(dim("  [Step 2 skipped]"))

    # ── Step 3: train model ───────────────────────────────────────────────────
    if not args.dashboard_only and not args.skip_training:
        step(3, "Train win prediction model")
        import os
        if not os.path.exists("training_data.csv"):
            print(red("  ✗ training_data.csv not found — skipping model training"))
        else:
            ok = run_step("win_predictor")
            if ok:
                print(green("  ✓ win_model.pkl + feature_importance.csv written"))
            else:
                print(yellow("  ⚠  Model training had errors — dashboard will show setup instructions"))
    else:
        print(dim("  [Step 3 skipped — using existing win_model.pkl]"))

    # ── Step 4: next-season forecast ─────────────────────────────────────────
    if not args.dashboard_only:
        step(4, "Generate next-season win forecast")
        import os
        if not os.path.exists("win_model.pkl"):
            print(yellow("  ⚠  win_model.pkl not found — skipping next-season forecast"))
        else:
            ok = run_step("forecast_next_season")
            if ok:
                print(green("  ✓ next_season_forecast.csv written"))
            else:
                print(yellow("  ⚠  Forecast had errors — Next Season tab will show setup instructions"))
    else:
        print(dim("  [Step 4 skipped]"))

    # ── Launch dashboard ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{bold('Pipeline complete')} {dim(f'({mins}m {secs}s)')}")
    print(bold("\n╔══════════════════════════════════════════════╗"))
    print(bold("║  Launching dashboard → http://127.0.0.1:8050 ║"))
    print(bold("╚══════════════════════════════════════════════╝\n"))

    # Import and run the dashboard (blocking)
    import dashboard
    dashboard.app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
