import torch.nn as nn

from s3prl import Output


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, output: Output):
        """
        This nn module is used as a placeholder for certain interfaces (e.g. the predictor in ssl tasks).

        Args:
            output (Output): the s3prk Output object

        Return:
            output (Output): exactly the same as input, the s3prk Output object
        """
        return output
