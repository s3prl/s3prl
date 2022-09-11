# Documentation

To auto-generate documents for S3PRL, please follow the following steps:

1. Activate an python env for the doc-generating tool to import all the modules to realize auto-documentation. There are various ways to achieve this. You can also follow:

```sh
conda create -y -n doc python=3.8
conda activate doc

cd $S3PRL_ROOT
pip install ".[dev]"
```

2. Auto-generate HTML files for all the packages, modules and their submodules listed in `$S3PRL_ROOT/valid_paths.txt`. The HTML files will appear in `$S3PRL_ROOT/docs/build/html`

```sh
cd $S3PRL_ROOT/docs_src
./rebuild_docs.sh
```

3. Launch the simple webserver to see the documentation.

```sh
cd $S3PRL_ROOT/docs/build/html
python3 -m http.server
```
