import os
import argparse
from pathlib import Path
from shutil import copyfile, copytree

parser = argparse.ArgumentParser()
parser.add_argument("--pr")
parser.add_argument("--sid")
parser.add_argument("--ks")
parser.add_argument("--ic")
parser.add_argument("--er_fold1")
parser.add_argument("--er_fold2")
parser.add_argument("--er_fold3")
parser.add_argument("--er_fold4")
parser.add_argument("--er_fold5")
parser.add_argument("--asr_no_lm")
parser.add_argument("--asr_with_lm")
parser.add_argument("--qbe")
parser.add_argument("--sf")
parser.add_argument("--sv")
parser.add_argument("--sd")
parser.add_argument("--se")
parser.add_argument("--st")
parser.add_argument("--ss")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

output_dir = Path(args.output_dir)
predict_dir = output_dir / "predict"
output_dir.mkdir(exist_ok=True)
predict_dir.mkdir(exist_ok=True)
processed_tasks = []

if args.pr is not None:
    task_name = "pr_public"
    processed_tasks.append(task_name)

    expdir = Path(args.pr)
    src = expdir / "test-hyp.ark"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.sid is not None:
    task_name = "sid_public"
    processed_tasks.append(task_name)

    expdir = Path(args.sid)
    src = expdir / "test_predict.txt"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.ks is not None:
    task_name = "ks_public"
    processed_tasks.append(task_name)

    expdir = Path(args.ks)
    src = expdir / "test_predict.txt"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.ic is not None:
    task_name = "ic_public"
    processed_tasks.append(task_name)

    expdir = Path(args.ic)
    src = expdir / "test_predict.csv"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.csv"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

er_processed = []
for foldid in range(1, 6):
    expdir = getattr(args, f"er_fold{foldid}")
    if expdir is not None:
        task_name = f"er_fold{foldid}_public"
        processed_tasks.append(task_name)
        er_processed.append(task_name)

        expdir = Path(expdir)
        src = expdir / f"test_fold{foldid}_predict.txt"
        assert src.is_file(), f"{src} not found"

        tgt_dir = predict_dir / task_name
        tgt_dir.mkdir(exist_ok=True)
        tgt = tgt_dir / "predict.txt"

        copyfile(src, tgt)
        print(f"{task_name} is included in the submission and will be scored after submitted.")

if len(er_processed) > 0 and len(er_processed) < 5:
        print(f"[Warning] - {er_processed} are included but only in {len(er_processed)} folds. er_public will NOT be scored. er_public will be scored only when all the 5 folds are submitted.")

if args.asr_no_lm is not None:
    task_name = "asr_public"
    processed_tasks.append(task_name)

    expdir = Path(args.asr_no_lm)
    src = expdir / f"test-clean-noLM-hyp.ark"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.asr_with_lm is not None:
    task_name = "asr_lm_public"
    processed_tasks.append(task_name)

    expdir = Path(args.asr_with_lm)
    src = expdir / f"test-clean-LM-hyp.ark"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.qbe is not None:
    task_name = "qbe_public"
    processed_tasks.append(task_name)

    expdir = Path(args.qbe)
    src = expdir / f"benchmark.stdlist.xml"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "benchmark.stdlist.xml"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.sf is not None:
    task_name = "sf_public"
    processed_tasks.append(task_name)

    expdir = Path(args.sf)
    src = expdir / "test-hyp.ark"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.sv is not None:
    task_name = "sv_public"
    processed_tasks.append(task_name)

    expdir = Path(args.sv)
    src = (expdir / "test_predict.txt").resolve()
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.sd is not None:
    task_name = "sd_public"
    processed_tasks.append(task_name)

    expdir = Path(args.sd)
    src_dir = expdir / "scoring" / "predictions"
    assert src_dir.is_dir(), f"{src_dir} not found"

    tgt_dir = predict_dir / task_name
    tgt_predict_dir = tgt_dir / "scoring" / "predictions"
    copytree(src_dir, tgt_predict_dir, dirs_exist_ok=True)

    upstream_rate = expdir / "frame_shift"
    if upstream_rate.is_file():
        copyfile(upstream_rate, tgt_dir / "frame_shift")

    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.se is not None:
    task_name = "se_public"
    processed_tasks.append(task_name)

    expdir = Path(args.se)
    src = (expdir / "test_metrics.txt").resolve()
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "metrics.txt"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.ss is not None:
    task_name = "ss_public"
    processed_tasks.append(task_name)

    expdir = Path(args.ss)
    src = (expdir / "test_metrics.txt").resolve()
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "metrics.txt"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

if args.st is not None:
    task_name = "st_public"
    processed_tasks.append(task_name)

    expdir = Path(args.st)
    src = expdir / "output-st-test.tsv"
    assert src.is_file(), f"{src} not found"

    tgt_dir = predict_dir / task_name
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.tsv"

    copyfile(src, tgt)
    print(f"{task_name} is included in the submission and will be scored after submitted.")

print("Zipping predictions for submission...")
os.chdir(output_dir)
os.system(f"zip -q -r predict.zip predict/")

print(f"Process {len(processed_tasks)} tasks: {' '.join(processed_tasks)}")