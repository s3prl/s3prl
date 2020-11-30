# Speech SSL Benchmarks

## How to run

### Start a new training experiment
- Example upstream + example downstream
```
python3 run_benchmark.py -m train -c benchmark/downstream/example/config.yaml -d example -u example -n HelloWorld
```

- Baseline upstream (fbank) + example downstream
```
python3 run_benchmark.py -m train -c benchmark/downstream/example/config.yaml -d example -u baseline -n BaselineFeature
```

### Resume training from a checkpoint
```
python3 run_benchmark.py -m train -e [ckpt]
```

### Test a checkpoint
```
python3 run_benchmark.py -m evaluate -e [ckpt]
```
