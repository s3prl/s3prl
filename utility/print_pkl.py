import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

with open(args.path, "rb") as f:
    pkl = pickle.load(f)
