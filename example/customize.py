import pandas as pd
import torch

from s3prl.problem import SuperbASR


class LowResourceLinearSuperbASR(SuperbASR):
    def prepare_data(
        self, prepare_data: dict, target_dir: str, cache_dir: str, get_path_only=False
    ):
        train_path, valid_path, test_paths = super().prepare_data(
            prepare_data, target_dir, cache_dir, get_path_only
        )

        # Take only the first 100 utterances for training
        df = pd.read_csv(train_path)
        df = df.iloc[:100]
        df.to_csv(train_path, index=False)

        return train_path, valid_path, test_paths

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, input_size, output_size) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(input_size, output_size)

            def forward(self, x, x_len):
                return self.linear(x), x_len

        return Model(downstream_input_size, downstream_output_size)


if __name__ == "__main__":
    LowResourceLinearSuperbASR().main()
