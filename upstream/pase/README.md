1. Modify the cuda version number of **cupy** in `requirements.txt`. The default `cupy-cuda102` presumes the cuda version of **10.2**. You should change `102` according to your cuda version which can be checked by `nvcc -V`.

2. `pip install -r requirements.txt`