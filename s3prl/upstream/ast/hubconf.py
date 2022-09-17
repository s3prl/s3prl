from .expert import UpstreamExpert
from s3prl.util.download import urls_to_filepaths


def ast(segment_secs: float = 10.24, refresh: bool = False, **kwds):
    ckpt = urls_to_filepaths(
        "https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1",
        refresh=refresh,
    )
    return UpstreamExpert(ckpt, segment_secs)
