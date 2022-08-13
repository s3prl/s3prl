# REDAME
## install sphinx
```shell
$ pip install -U sphinx
```

## set up
at s3prl root directory (s3prl-private/)
```shell
$ mkdir docs
$ cd docs
$ sphinx-quickstart
$ cd ..
$ sphinx-apidoc -d 3 --separate --implicit-namespace -o docs ./s3prl s3prl/downstream s3prl/interface s3prl/preprocess s3prl/pretrain s3prl/problem s3prl/sampler s3prl/submit s3prl/superb s3prl/upstream s3prl/utility s3prl/wrapper s3prl/__init__.py s3prl/hub.py s3prl/optimizers.py s3prl/run_downstream.py s3prl/run_pretrain.py s3prl/run_while.sh s3prl/schedulers.py
```

## install dependencies
```shell
$ cd s3prl-private/docs
$ echo "furo
torch
numpy
pandas
tqdm
pytorch_lightning
matplotlib
ipdb>=0.13.9
PyYAML
transformers
torchaudio
gdown
sklearn
joblib
tensorboardX
librosa
scipy
lxml
h5py
dtw
catalyst
sox
six
easydict
Resemblyzer
sentencepiece
pysoundfile
asteroid
sacrebleu
speechbrain
omegaconf
editdistance" > requirement.txt
$ pip install -r requirement.txt
```

add custom.js at s3prl-private/docs/_static/js/
(just paste the following lines)
```javascript=
/*
change the default sphinx.ext.linkcode's [source] to [Github]
*/
document.querySelectorAll(".reference.external .viewcode-link .pre").forEach(item => {
    item.innerHTML = "[Github]"
    item.style.marginRight = "3px"
})

```

modify s3prl-private/docs/index.rst
```diff
# remove these lines
- .. toctree::
-    :maxdepth: 2
-    :caption: Contents:

# replaced with this line
+ .. include:: s3prl.rst
```

modify s3prl-private/docs/conf.py
```python
# add these lines at top
import inspect
import os
import sys
for x in os.walk('..'):
	sys.path.insert(0, x[0])
    
# add extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.linkcode'
]

html_js_files = [
    'js/custom.js',
]

def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info['module']]
        if info['fullname'] == 'InitConfig.args':	return None	
        if info['fullname'] == 'InitConfig.kwargs':	return None	
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part) 

        if isinstance(obj, property):	return None	

        fn = inspect.getsourcefile(obj)	
        fn = os.path.relpath(fn, start=os.path.dirname(os.path.abspath(__file__))[:-4])

        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:	return None

    tag = 'master' if 'dev' in release else ('v' + release)		# s3prl github version

    try:
        filename = '%s#L%d-L%d' % find_source()			# specify file page with line number
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'		# cannot find corresponding codeblock, use the file page instead

    return "https://github.com/s3prl/s3prl-private/blob/%s/%s" % (tag, filename)
```

to use the furo theme, add this line in s3prl-private/docs/conf.py (replace the original alabaster theme)
```python
html_theme = "furo"
```

## generate html files
at s3prl-private/docs/
```shell
$ make html
```

the html files will be generated at **s3prl-private/docs/_build/html/**
click on **index.html** to view the doc page on your browser

if you want to see how your modified codes looks like, simply do
```shell
$ make clean html        # this remove the old html files
$ make html              # generate new html files
```