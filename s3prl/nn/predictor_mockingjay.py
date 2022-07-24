from torch import nn

from s3prl import Output
from s3prl.nn.transformer_mockingjay import (
    ACT2FN,
    TransformerConfig,
    TransformerLayerNorm,
)


class PredictorMockingjay(nn.Module):
    """
    The predictor model for SSL pre-training tasks.
    Currently supporting SSL problems of Mockingjay, Tera, and Audio Albert.
    """

    def __init__(self, config, output_dim, input_dim=None, **kwargs):
        """
        Args:
            config (TransformerConfig):
                A `TransformerConfig` class instance with the configuration to build a new model,
                can also be a `dict` that initializes the TransformerConfig class
            output_dim (int):
                The output dimension of predictor
            input_dim (int):
                The input dimension of predictor, if `None` is given, then use the `hidden_size` defined in `config`.
                Default: None
        """

        super(PredictorMockingjay, self).__init__()
        if type(config) is dict:
            config = TransformerConfig(**config)
        self.output_size = output_dim
        if input_dim is None:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(input_dim, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = TransformerLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output = nn.Linear(config.hidden_size, self.output_size)

    def forward(self, inputs, output_states=False):
        """
        Args:
            inputs (torch.LongTensor):
                A torch.LongTensor of shape [batch_size, sequence_length, input_dim]
            output_states (bool):
                A boolean which controls whether to return the `hidden_states` of the predictor.
                Default: False
        Return:
            Output (s3prl.Output):
                An Output module that contains `prediction` and/or `hidden_states`.
        """
        hidden_states = inputs.hidden_states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction = self.output(hidden_states)
        if output_states:
            return Output(hidden_states=hidden_states, prediction=prediction)
        else:
            return Output(prediction=prediction)
