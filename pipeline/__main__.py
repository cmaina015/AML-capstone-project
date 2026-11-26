"""Command line for the pipeline.

    python -m pipeline run     --data-dir PATH [--config configs/baseline.json] [--out-dir artifacts]
    python -m pipeline ingest  --data-dir PATH       # stage 1: index + integrity checks
    python -m pipeline split                         # stage 2: patient-level folds + leakage audit
    python -m pipeline train                         # stage 3: CV, baseline, final model (resumable)
    python -m pipeline export                        # stage 4: deployable artifacts + reload check
    python -m pipeline predict IMAGE [IMAGE ...] [--artifacts deployment_artifacts] [--heatmap-dir DIR]

Every stage reads what the previous one wrote to --work-dir, so stages can be re-run on their own.
"""
import argparse
import logging
import sys
import time

from .config import Config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON file overriding the defaults (see configs/)")
    p.add_argument("--data-dir", help="BreakHis folder or zip")
    p.add_argument("--work-dir", help="intermediate files (default: work)")
    p.add_argument("--out-dir", help="exported artifacts (default: artifacts)")
    p.add_argument("--run-name", help="name for this configuration's results under work/runs/")
    p.add_argument("--splits-csv", help="reuse the folds of an existing split_assignments.csv")
    p.add_argument("--epochs", type=int, help="override hp.epochs (e.g. 1 for a quick smoke run)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--no-baseline", action="store_true", help="skip the leaky image-level baseline")
    p.add_argument("--no-shift-eval", action="store_true", help="skip the other-magnification test")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")


def _config(args) -> Config:
    cfg = Config.load(args.config, data_dir=args.data_dir, work_dir=args.work_dir, out_dir=args.out_dir,
                      run_name=args.run_name, splits_csv=args.splits_csv, batch_size=args.batch_size,
                      num_workers=args.num_workers)
    if args.epochs:
        cfg.hp["epochs"] = args.epochs
    if args.no_baseline:
        cfg.run_leakage_baseline = False
    if args.no_shift_eval:
        cfg.run_shift_eval = False
    if args.no_pretrained:
        cfg.pretrained = False
    return cfg


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "ingest", "split", "train", "export"):
        _common(sub.add_parser(name))
    pr = sub.add_parser("predict")
    pr.add_argument("images", nargs="+")
    pr.add_argument("--artifacts", default="deployment_artifacts")
    pr.add_argument("--heatmap-dir")
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        from . import predict
        predict.run(args.images, args.artifacts, args.heatmap_dir)
        return 0

    cfg = _config(args)
    t0 = time.time()
    if args.cmd in ("run", "ingest"):
        if not cfg.data_dir:
            ap.error("--data-dir is required for ingest/run")
        from . import ingest
        ingest.run(cfg)
    if args.cmd in ("run", "split"):
        from . import split
        split.run(cfg)
    if args.cmd in ("run", "train", "export"):
        from . import train
        trained = train.run(cfg)       # finished folds/final model are loaded, not retrained
        if args.cmd in ("run", "export"):
            from . import export
            export.run(cfg, trained)
    logging.getLogger("pipeline").info("Done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
