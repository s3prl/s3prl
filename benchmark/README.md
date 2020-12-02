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

- 相關檔案除了根目錄的 `run_benchmark.py` 外，全放在 `benchmark/` 資料夾。
- 如何加新的 upstream 或 downstream 都有 example code，放在 `benchmark/upstream/example/` 和 `benchmark/downstream/example/` 中。
- 各自 `example/` 下的 `expert.py` 定義了 upstream/downstream 各自該 implement 的 interface functions。

### Upstream

- 每個不同的 pre-trained method 要各自有一個資料夾放在 `benchmark/upstream/` 中，以 pre-trained method **名稱** 命名資料夾，此**名稱**會直接等於 command-line args 要下的 --upstream **名稱**。
    - eg. `benchmark/upstream/apc/` ⇒ `--upstream apc`
- 整個 training pipeline 只會 imports an `UpstreamExpert` class defined in `benchmark/upstream/[--upstream]/expert.py`。
- The filename `expert.py` and the classname `UpstreamExpert` can **NOT** be modified.

### Downstream

- 每個不同的 pre-trained method 要各自有一個資料夾放在 `benchmark/downstream/` 中，以 downstream **名稱** 命名資料夾，此**名稱**會直接等於 command-line args 要下的 --downstream **名稱**。
    - eg. `benchmark/downstream/phone/` ⇒ `--downstream phone`
- 整個 training pipeline 只會 imports an `DownstreamExpert` class defined in `benchmark/downstream/[--downstream]/expert.py`。
- The filename `expert.py` and the classname `UpstreamExpert` can **NOT** be modified.

# How to add new downstream

有三件事情需要完成：
- Code implementation
- Documentation
- Run experiments

## Code implementation
### Workflow

以小明準備來做 phone classification 為例。

- 小明先在 `benchmark/downstream/` 之下開一個 phone 資料夾。
- 此 `benchmark/downstream/phone/` 資料夾完全歸小明所有，其他人都不應該動此資料夾中的內容。
- 另一方面，小明也只能在這個資料夾中做事情，不應該動到資料夾以外的內容。
- 小明複製了一份 `benchmark/downstream/example/expert.py` 到自己的資料夾下，不更改 `expert.py` filename 與 `DownstreamExpert` classname。
- 根據 `DownstreamExpert` 中定義的 interface functions，開始 implement 專屬於 phone classification 的 interface functions。(只要 interface 有完成即可，其他 coding style 不需要完全 follow `example/` 中的結構。)
- 主線分支：
    - 小明 implement 完，也成功測試完下方三個指令：`train`, `resume training`, `test`
        - **⇒** 完成 coding 任務準備來寫文件。
    - 小明 implement 到一半發現他一定需要改資料夾以外的 code 才能繼續做事 
        - **⇒** 聯絡 Leo 趕緊來討論要怎麼修改。

```bash
# train
python3 run_benchmark.py -m train -n PhoneFbank -u baseline -d phone -c benchmark/downstream/phone/config.yaml

# resume training
python3 run_benchmark.py -m train -e [ckpt]

# test
python3 run_benchmark.py -m evaluate -e [ckpt]
```

### Push commits

- 因為每個 task 可以動的資料夾完全獨立，大家都可以直接 push 到 benchmark branch 而不用擔心 conflict。
- 請也只 push 到 benchmark branch，不要動到其他 branch。

## Documentation
- 先以我們這個 team 的成員讀完能知道要怎麼從無到有跑出合理數字為目標，未來 Leo 會再修成面向所有使用者的樣子。
- 各個 task 請寫在上方 `Downstream documentation` 的相對應欄位，歡迎在該欄位上標記任何你想要的個人資訊。

## Run experiments
To be discussed.
