"""
Parse the QUESST14 corpus

Authors:
  * Leo 2022
  * Cheng Liang 2022
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "Quesst14",
]


class Quesst14:
    def __init__(self, dataset_root: str):
        dataset_root = Path(dataset_root)
        self.doc_paths = self._english_audio_paths(
            dataset_root, "language_key_utterances.lst"
        )
        self.dev_query_paths = self._english_audio_paths(
            dataset_root, f"language_key_dev.lst"
        )
        self.eval_query_paths = self._english_audio_paths(
            dataset_root, f"language_key_eval.lst"
        )

        self.n_dev_queries = len(self.dev_query_paths)
        self.n_eval_queries = len(self.eval_query_paths)
        self.n_docs = len(self.doc_paths)

    @staticmethod
    def _english_audio_paths(dataset_root_path, lst_name):
        """Extract English audio paths."""
        audio_paths = []

        with open(dataset_root_path / "scoring" / lst_name) as f:
            for line in f:
                audio_path, lang = tuple(line.strip().split())
                if lang != "nnenglish":
                    continue
                audio_path = re.sub(r"^.*?\/", "", audio_path)
                audio_paths.append(dataset_root_path / audio_path)

        return audio_paths

    @property
    def valid_queries(self):
        return self.dev_query_paths

    @property
    def test_queries(self):
        return self.eval_query_paths

    @property
    def docs(self):
        """
        Valid and Test share the same document database
        """
        return self.doc_paths

    @classmethod
    def download_dataset(cls, tgt_dir: str) -> None:
        import os
        import tarfile

        import requests

        assert os.path.exists(
            os.path.abspath(tgt_dir)
        ), "Target directory does not exist"

        def unzip_targz_then_delete(filepath: str):
            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(tgt_dir))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logger.info(f"Saving {filename} to", os.path.abspath(filepath))
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logger.info(f"{filename} successfully downloaded")
                unzip_targz_then_delete(filepath)
            else:
                logger.info(f"Download failed: status code {r.status_code}\n{r.text}")

        if not os.path.exists(
            os.path.join(os.path.abspath(tgt_dir), "quesst14Database/")
        ):
            download_from_url("https://speech.fit.vutbr.cz/files/quesst14Database.tgz")
        logger.info(
            f"Quesst14 dataset downloaded. Located at {os.path.abspath(tgt_dir)}/quesst14Database/"
        )
