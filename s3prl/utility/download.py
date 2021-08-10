import os
import hashlib
from filelock import FileLock

import torch
import gdown


def _download(filename, url, refresh, agent):
    dirpath = f'{torch.hub.get_dir()}/s3prl_cache'
    os.makedirs(dirpath, exist_ok=True)
    filepath = f'{dirpath}/{filename}'
    with FileLock(filepath + ".lock"):
        if not os.path.isfile(filepath) or refresh:
            if agent == 'wget':
                os.system(f'wget {url} -O {filepath}')
            elif agent == 'gdown':
                gdown.download(url, filepath, use_cookies=False)
            else:
                print('[Download] - Unknown download agent. Only \'wget\' and \'gdown\' are supported.')
                raise NotImplementedError
        else:
            print(f'Using cache found in {filepath}\nfor {url}')
    return filepath


def _urls_to_filepaths(*args, refresh=False, agent='wget'):
    """
    Preprocess the URL specified in *args into local file paths after downloading

    Args:
        Any number of URLs (1 ~ any)

    Return:
        Same number of downloaded file paths
    """

    def url_to_filename(url):
        assert type(url) is str
        m = hashlib.sha256()
        m.update(str.encode(url))
        return str(m.hexdigest())

    def url_to_path(url, refresh):
        if type(url) is str and len(url) > 0:
            return _download(url_to_filename(url), url, refresh, agent=agent)
        else:
            return None

    paths = [url_to_path(url, refresh) for url in args]
    return paths if len(paths) > 1 else paths[0]


def _gdriveids_to_filepaths(*args, refresh=False):
    """
    Preprocess the Google Drive id specified in *args into local file paths after downloading

    Args:
        Any number of Google Drive ids (1 ~ any)

    Return:
        Same number of downloaded file paths
    """

    def gdriveid_to_url(gdriveid):
        if type(gdriveid) is str and len(gdriveid) > 0:
            return f'https://drive.google.com/uc?id={gdriveid}'
        else:
            return None

    return _urls_to_filepaths(*[gdriveid_to_url(gid) for gid in args], refresh=refresh, agent='gdown')
