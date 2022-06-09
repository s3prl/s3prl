import re
import tempfile

from s3prl.base.fileio import RTTMHandler
from s3prl.base.workspace import Workspace


def test_rttm():
    with tempfile.TemporaryDirectory() as path:
        rttm = {
            "reco1": {
                "spk1": [(1, 3), (2, 4)],
                "spk2": [(2, 5)],
            },
        }
        work = Workspace(path)
        work.put(rttm, "tmp", "rttm")
        new_rttm = work["tmp"]
        assert rttm == new_rttm

        with open(work / "tmp.rttm") as file:
            first_line = file.readline().strip()
            assert (
                re.search("SPEAKER (.+) 1 (.+) (.+) <NA> <NA> (.+) <NA>", first_line)
                is not None
            )
