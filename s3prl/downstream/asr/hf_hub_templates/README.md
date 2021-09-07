---
library_name: superb
benchmark: superb
task: asr
datasets:
- superb
tags:
- automatic-speech-recognition
- ${upstream_model}
widget:
- label: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
---

# Fine-tuned s3prl model for ASR