import csv
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from torch.utils.data import Dataset


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        bucket_size,
        path,
        ascending=False,
        ratio=1.0,
        offset=0,
        **kwargs,
    ):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        for s in split:
            with open(s, "r") as fp:
                rows = csv.reader(fp, delimiter="\t")
                file_list, text = [], []
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    file_list.append(join(path, row[0]))
                    text.append(tokenizer.encode(row[1]))

                print(f"Found {len(file_list)} samples.")

        if ratio < 1.0:
            print(f"Ratio = {ratio}, offset = {offset}")
            skip = int(1.0 / ratio)
            file_list, text = file_list[offset::skip], text[offset::skip]
            total_len = 0.0
            for f in file_list:
                total_len += getsize(f) / 32000.0
            print(
                "Total audio len = {:.2f} mins = {:.2f} hours".format(
                    total_len / 60.0, total_len / 3600.0
                )
            )

        self.file_list, self.text = file_list, text

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list) - self.bucket_size, index)
            return [
                (f_path, txt)
                for f_path, txt in zip(
                    self.file_list[index : index + self.bucket_size],
                    self.text[index : index + self.bucket_size],
                )
            ]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)
