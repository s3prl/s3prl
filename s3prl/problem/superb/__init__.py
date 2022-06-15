"""
This is the official SUPERB package.
Compared to the old SUPERB codebase under **s3prl/downstream**, there are several improvements.

.. note::

    - The old codebase is more recipe-like. It entangles lots of logic together in a single God class
      (DownstreamExpert) to solve a task, making it hard to grap a single component from it and reuse.

    - The God class entangles the corpus parsing, which is real data dependent. This means it is
      really hard for the old codebase to do unit-test since you need the real dataset downloaded
      to test a task, which is very time-consuming, and it becomes complicated to test just a single
      corpus-independent component.

    - The God class entangles the criterion with the model without a clean boundary, making it hard
      to replace a random downstream model.

    - The pre-trained model preparation is hided deeply in the training logic, while it should be exposed at
      the very outside. Since benchmarking any custom pre-trained model is the main purpose of SUPERB instead
      of just the pre-trained models available in S3PRL.

    - The old codebase is more recipe-like. The user must clone the repository to do any thing with S3PRL,
      which limit its to have broader impacts.
"""

from s3prl.problem.superb.base import SuperbProblem

from .asr import SuperbASR
from .er import SuperbER
from .ic import SuperbIC
from .ks import SuperbKS
from .pr import SuperbPR
from .sid import SuperbSID
from .sv import SuperbSV
