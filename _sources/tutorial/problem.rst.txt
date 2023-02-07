Use Problem module to run customizable recipes
=======================================================

The :obj:`s3prl.problem` module provides customizable recipes in pure python (almost).
See :obj:`s3prl.problem` for all the recipes ready to be ran.


Usage 1. Import and run on Colab
--------------------------------

All the problem class follows the same usage

    >>> import torch
    >>> from s3prl.problem import SuperbASR
    ...
    >>> problem = SuperbASR()
    >>> config = problem.default_config()
    >>> print(config)
    ...
    >>> # See the config for the '???' required fields and fill them
    >>> config["target_dir"] = "result/asr_exp"
    >>> config["prepare_data"]["dataset_root"] = "/corpus/LibriSpeech/"
    ...
    >>> problem.run(**config)


Usage 2. Run & configure from CLI
-----------------------------------

If you want to directly run from command-line, write a python script (:code:`asr.py`) as follow:

.. code-block::

    # This is asr.py

    from s3prl.problem import SuperbASR
    SuperbASR().main()

Then, run the command below:

    >>> # Note that the main function supports overridding a field in the config by:
    >>> #   --{field_name} {value}
    >>> #   --{outer_field_name}.{inner_field_name} {value}
    ...
    >>> python3 asr.py --target_dir result/asr_exp --prepare_data.dataset_root /corpus/LibriSpeech/


Usage 3. Run & configure with the unified :obj:`s3prl-main`
-----------------------------------------------------------

However, this means that for every problem you still need to create a file.
Hence, we provide an easy helper supporting all the problems in :obj:`s3prl.problem`:

    >>> python3 -m s3prl.main SuperbASR --target_dir result/asr_exp --prepare_data.dataset_root /corpus/LibriSpeech/

or use our CLI entry: :code:`s3prl-main`

    >>> s3prl-main SuperbASR --target_dir result/asr_exp --prepare_data.dataset_root /corpus/LibriSpeech/

Customization
-------------

The core feature of the :obj:`s3prl.problem` module is customization.
You can easily change the corpus, change the SSL upstream model, change the downstream model,
optimizer, scheduler... etc, which can all be freely defined by you!

We demonstrate how to change the corpus and the downstream model in the following :code:`new_asr.py`:

.. code-block:: python

    # This is new_asr.py

    import torch
    import pandas as pd
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
            class Model(torch.nn.Module):
                def __init__(self, input_size, output_size) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(input_size, output_size)

                def forward(self, x, x_len):
                    return self.linear(x), x_len

            return Model(downstream_input_size, downstream_output_size)


    if __name__ == "__main__":
        LowResourceLinearSuperbASR().main()


By subclassing :obj:`SuperbASR`, we create a new problem called :code:`LowResourceLinearSuperbASR` by
overridding the :code:`prepare_data` and :code:`build_downstream` methods. After this simple modification,
now the :code:`LowResourceLinearSuperbASR` works exactly the same as :code:`SuperbASR` while with two slight
setting changes, and then you can follow the first two usages introduced above to launch this new class.

For example:

    >>> python3 new_asr.py --target_dir result/new_asr_exp --prepare_data.dataset_root /corpus/LibriSpeech/
