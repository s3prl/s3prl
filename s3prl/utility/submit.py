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
parser.add_argument("--asr_nolm")
parser.add_argument("--asr_lm")
parser.add_argument("--qbe")
parser.add_argument("--sf")
parser.add_argument("--sv")
parser.add_argument("--sd")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

output_dir = Path(args.output_dir)
predict_dir = output_dir / "predict"
output_dir.mkdir(exist_ok=True)
predict_dir.mkdir(exist_ok=True)
processed_tasks = []

if args.pr is not None:
    processed_tasks.append("pr")

    expdir = Path(args.pr)
    src = expdir / "test-hyp.ark"
    assert src.is_file()

    tgt_dir = predict_dir / "pr_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)

if args.sid is not None:
    processed_tasks.append("sid")

    expdir = Path(args.sid)
    src = expdir / "test_predict.txt"
    assert src.is_file()

    tgt_dir = predict_dir / "sid_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)

if args.ks is not None:
    processed_tasks.append("ks")

    expdir = Path(args.ks)
    src = expdir / "test_predict.txt"
    assert src.is_file()

    tgt_dir = predict_dir / "ks_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)

if args.ic is not None:
    processed_tasks.append("ic")

    expdir = Path(args.ic)
    src = expdir / "test_predict.csv"
    assert src.is_file()

    tgt_dir = predict_dir / "ic_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.csv"

    copyfile(src, tgt)

if args.er_fold1 is not None:
    processed_tasks.append("er")

    predictions = []
    for foldid in range(1, 6):
        expdir = Path(getattr(args, f"er_fold{foldid}"))
        assert expdir.is_dir()
        src = expdir / f"test_fold{foldid}_predict.txt"
        assert src.is_file()
        predictions.append(str(src))
    
    tgt_dir = predict_dir / "er_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"
    os.system(f"cat {' '.join(predictions)} > {tgt.resolve()}")

if args.asr_nolm is not None:
    processed_tasks.append("asr-noLM")

    expdir = Path(args.asr_nolm)
    src = expdir / f"test-clean-noLM-hyp.ark"
    assert src.is_file()

    tgt_dir = predict_dir / "asr_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)

if args.asr_lm is not None:
    processed_tasks.append("asr-LM")

    expdir = Path(args.asr_lm)
    src = expdir / f"test-clean-LM-hyp.ark"
    assert src.is_file()

    tgt_dir = predict_dir / "asr_lm_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)

if args.qbe is not None:
    processed_tasks.append("qbe")

    expdir = Path(args.qbe)
    src = expdir / f"benchmark.stdlist.xml"
    assert src.is_file()

    tgt_dir = predict_dir / "qbe_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "benchmark.stdlist.xml"

    copyfile(src, tgt)

if args.sf is not None:
    processed_tasks.append("sf")

    expdir = Path(args.sf)
    src = expdir / "test-hyp.ark"
    assert src.is_file()

    tgt_dir = predict_dir / "sf_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.ark"

    copyfile(src, tgt)

if args.sv is not None:
    processed_tasks.append("sv")

    expdir = Path(args.sv)
    src = expdir / "test_predict.txt"
    assert src.is_file()

    tgt_dir = predict_dir / "sv_public"
    tgt_dir.mkdir(exist_ok=True)
    tgt = tgt_dir / "predict.txt"

    copyfile(src, tgt)

if args.sd is not None:
    processed_tasks.append("sd")

    expdir = Path(args.sd)
    src_dir = expdir / "scoring" / "predictions"
    assert src_dir.is_dir()

    tgt_dir = predict_dir / "sd_public"
    copytree(src_dir, tgt_dir, dirs_exist_ok=True)

os.chdir(output_dir)
os.system(f"zip -r predict.zip predict/")

print(f"Process {len(processed_tasks)} tasks: {' '.join(processed_tasks)}")