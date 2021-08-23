# Register customized upstream

Only the forward functions in the class [UpstreamExpert](./expert.py) are required to implement. One can refer to [upstream/example/expert.py](./expert.py) and [upstream/example/hubconf.py](./hubconf.py) for the minimal implementation and use the following command to help the development

```bash
python3 run_downstream.py -m train -n HelloWorld -u customized_upstream -d example
```