import os
import argparse
from typing import List
from pathlib import Path
from scipy.stats import ttest_rel


def find_scoring_result_files(folder: str):
    assert Path(folder).is_dir()
    all_filenames = sorted(os.listdir(folder))
    result_files = []
    for filename in all_filenames:
        filepath: Path = Path(folder) / filename
        if filepath.is_file() and filename.startswith("result"):
            result_files.append(filepath)
    return result_files


def read_scoring_file(filepath: str):
    filepath: Path = Path(filepath)
    assert filepath.is_file()

    name2der = {}
    with filepath.open() as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines[2:-1]:
            fields = line.split()
            assert len(fields) == 16
            name, der = fields[:2]
            assert name not in name2der
            name2der[name] = float(der)

        fields = lines[-1].split()
        assert fields[0] == "***"
        assert fields[1] == "OVERALL"
        assert fields[2] == "***"
        overall_der = float(fields[3])

    return overall_der, name2der


def find_and_read_best_scoring_result(expdir: str):
    scoring_dir: Path = Path(expdir) / "scoring"
    assert scoring_dir.is_dir()

    score_files: List[Path] = find_scoring_result_files(scoring_dir)
    results = []
    for score_file in score_files:
        overall_der, name2der = read_scoring_file(score_file)
        results.append((score_file, overall_der, name2der))
    results.sort(key=lambda x: x[1])
    best_score_file, best_overall_der, best_name2der = results[0]
    return best_score_file, best_overall_der, best_name2der


def pairwise_t_test_pvalue(metrics1: List[float], metrics2: List[float]) -> float:
    stats = ttest_rel(metrics1, metrics2, nan_policy="raise")
    return stats.pvalue


def form_paired_data(name2value1: List[float], name2value2: List[float]) -> float:
    names = sorted(list(name2value1.keys()))
    values1 = [name2value1[name] for name in names]
    values2 = [name2value2[name] for name in names]
    return values1, values2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir1")
    parser.add_argument("expdir2")
    args = parser.parse_args()

    score_file1, overall_der1, name2der1 = find_and_read_best_scoring_result(
        args.expdir1
    )
    score_file2, overall_der2, name2der2 = find_and_read_best_scoring_result(
        args.expdir2
    )

    pvalue = pairwise_t_test_pvalue(*form_paired_data(name2der1, name2der2))

    print(f"{score_file1} DER: {overall_der1}")
    print(f"{score_file2} DER: {overall_der2}")
    print(f"pvalue: {pvalue}")
