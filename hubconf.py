# -*- coding: utf-8 -*- #
"""************************************************************************************************
   FileName     [ hubconf.py ]
   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
   Author       [ S3PRL ]
   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
************************************************************************************************"""

import os
import hashlib
import importlib

import torch
import gdown
dependencies = ['torch', 'gdown']


def _gdown(filename, url, refresh):
    filepath = f'{torch.hub.get_dir()}/{filename}'
    if not os.path.isfile(filepath) or refresh:
        print(f'Downloading file to {filepath}')
        gdown.download(url, filepath, use_cookies=False)
    else:
        print(f'Using cache found in {filepath}')
    return filepath


def _urls_to_filepaths(*args, refresh=False):
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
            return _gdown(url_to_filename(url), url, refresh)
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

    return _urls_to_filepaths(*[gdriveid_to_url(gid) for gid in args], refresh=refresh)


for upstream_dir in os.listdir('benchmark/upstream'):
    hubconf_path = os.path.join('benchmark/upstream', upstream_dir, 'hubconf.py')
    if os.path.isfile(hubconf_path):
        module_path = f'benchmark.upstream.{upstream_dir}.hubconf'
        _module = importlib.import_module(module_path)
        for variable_name in dir(_module):
            _variable = getattr(_module, variable_name)
            if callable(_variable) and variable_name[0] != '_':
                globals()[variable_name] = _variable
