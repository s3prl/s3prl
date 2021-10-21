# Register customized upstream

One can refer to [`upstream/example/expert.py`](./expert.py) and [`upstream/example/hubconf.py`](./hubconf.py) for the minimal implementation and use the following command to help the development.

```bash
python3 run_downstream.py -m train -n HelloWorld \
    -u customized_upstream \
    -k PATH_OF_CHECKPOINT \
    -g PATH_OF_CONFIG \
    -d example
```

Note that each function name in `hubconf.py` will be the `Name` of an upstream, instead the name of the folder containing these files. This `Name` can then be specified by `-u` in `run_downstream.py`
