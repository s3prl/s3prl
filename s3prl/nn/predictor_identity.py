import torch.nn as nn

from s3prl import Output


class PredictorIdentity(nn.Module):
    """
    This nn module is used as a predictor placeholder for certain SSL problems.
    """

    def __init__(self, **kwargs):
        super(PredictorIdentity, self).__init__()

    def forward(self, output: Output):
        """
        Args:
            output (s3prl.Output): An Output module

        Return:
            output (s3prl.Output): exactly the same as input, an Output module
        """
        return output
