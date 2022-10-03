import tempfile

import torch
import torchaudio

SAMPLE_RATE = 16000


def test_sox_effect():
    effects = [
        ["channels", "1"],
        ["rate", "16000"],
        ["gain", "-3.0"],
    ]

    with tempfile.NamedTemporaryFile() as file:
        tensor = torch.randn(1, 16000 * 10)
        filename = f"{file.name}.wav"
        torchaudio.save(filename, tensor, SAMPLE_RATE)
        wav1, sr1 = torchaudio.sox_effects.apply_effects_file(filename, effects=effects)
        wav2, sr2 = torchaudio.sox_effects.apply_effects_tensor(
            tensor, SAMPLE_RATE, effects
        )
        torch.allclose(wav1, wav2)
        assert sr1 == sr2
