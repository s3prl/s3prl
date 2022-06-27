import logging
import re
import logging
from pathlib import Path

from s3prl import Container
from s3prl.util import registry

from .base import Corpus

logger = logging.getLogger(__name__)


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


@registry.put()
def quesst14_for_qbe(dataset_root: str):
    corpus = Quesst14(dataset_root)

    def path_to_dict(path: str):
        return dict(
            wav_path=path,
        )

    return Container(
        all_data={
            Path(path).stem: path_to_dict(path)
            for path in (corpus.valid_queries + corpus.test_queries + corpus.docs)
        },
        valid_keys=[Path(path).stem for path in corpus.valid_queries],
        test_keys=[Path(path).stem for path in corpus.test_queries],
        doc_keys=[Path(path).stem for path in corpus.docs],
    )
