from s3prl.nn import UpstreamDownstreamModel
from s3prl.nn.upstream_downstream_model import UpstreamExample, DownstreamExample

def test_upstream_downstream():
    upstream = UpstreamExample()
    downstream = DownstreamExample()
    model = UpstreamDownstreamModel(upstream, downstream)
