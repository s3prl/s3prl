from s3prl.downstream.runner import Runner
from typing import Dict
import torch
import os


class PreTrainedModel(Runner):
    def __init__(self, path=""):
        """
        Initialize downstream model.
        """
        ckp_file = os.path.join(path, "model.ckpt")
        ckp = torch.load(ckp_file, map_location='cpu')
        ckp["Args"].init_ckpt = ckp_file
        ckp["Args"].mode = "inference"
        ckp["Args"].device = "cpu"

        Runner.__init__(self, ckp["Args"], ckp["Config"])

    def __call__(self, inputs)-> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`dict`:. The object should return a dictionary like
            {"frames": "XXX"} which contains the frames where one, both, or none
            of the speakers are speaking.
        """
        for entry in self.all_entries:
            entry.model.eval()

        inputs = [torch.FloatTensor(inputs)]

        with torch.no_grad():
            features = self.upstream.model(inputs)
            features = self.featurizer.model(inputs, features)
            preds = self.downstream.model.inference(features, [])
        return {"frames": preds[0]}
