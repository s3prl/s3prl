from s3prl import hub

for _option in hub.options():
    globals()[_option] = getattr(hub, _option)
