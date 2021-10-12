import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--tgt", required=True)
args = parser.parse_args()

def read_file(path, callback=lambda x: x, sep=" ", default_value=""):
    content = {}
    with open(path, "r") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            fields = line.strip().split(sep, maxsplit=1)
            if len(fields) > 1:
                filename, value = fields
            else:
                filename = fields[0]
                value = default_value
            content[filename] = callback(value)
    return content

src = read_file(args.src)
tgt = read_file(args.tgt)

match = []
for key in src.keys():
    truth = tgt[key]
    predict = src[key]
    if truth == predict:
        match.append(1)
    else:
        match.append(0)

print(sum(match) / len(match))