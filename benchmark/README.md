# Speech SSL Benchmark


# Minimum run-up
Using the following commands, you can always run the benchmark codes without upstream/downstream dependency: checkpoints, dataset, etc...

### Start a new downstream training experiment
- Example (pseudo) upstream + example downstream
```bash
python3 run_benchmark.py -m train -n HelloWorld -u example -d example -c benchmark/downstream/example/config.yaml
```

- Baseline (fbank) upstream + example downstream
```bash
python3 run_benchmark.py -m train -n BaselineFeature -u baseline -d example -c benchmark/downstream/example/config.yaml
```

### Resume training from a checkpoint
```bash
python3 run_benchmark.py -m train -e [ckpt]
```

### Test a checkpoint
```bash
python3 run_benchmark.py -m evaluate -e [ckpt]
```

# Upstream documentation
### CPC
```
To be discussed.
```

### PaseNet
```
To be discussed.
```

### Mockingjay
```
To be discussed.
```

### Tera
```
To be discussed.
```

### Audio Albert
```
To be discussed.
```

### APC
```
To be discussed.
```

### VQ-APC
```
To be discussed.
```

### NPC
```
To be discussed.
```

### wav2vec
```
To be discussed.
```

### vq-wav2vec
```
To be discussed.
```

### wav2vec 2.0
```
To be discussed.
```

# Downstream documentation
## Content

### Phone Classification
```
To be discussed.
```

### Spoken Term Detection
```
To be discussed.
```

### Automatic Speech Recognition
```
To be discussed.
```

## Speaker

### Speaker Recognition
```
To be discussed.
```

### Speaker Verification
```
To be discussed.
```

## Paralinguistic

### Emotion Classification
```
To be discussed.
```

## Semantic

### Intent Classification
```
To be discussed.
```

### Question Answering
```
To be discussed.
```

# Framework design
[Chinese version](https://hackmd.io/@QMLdEc5PRayZZIfBA3H1kA/BJQW_l8jD)

# How to add new downstream
[Chinese version](https://hackmd.io/@QMLdEc5PRayZZIfBA3H1kA/BJQW_l8jD)
