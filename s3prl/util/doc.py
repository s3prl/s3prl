import yaml

from s3prl.base.container import Container


def _longestCommonPrefix(strs):
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def autodoc_attr(*fields):
    def _document_attributes(obj):
        doc = obj.__doc__ or ""

        lines = [line for line in doc.split("\n") if len(line) > 0]
        indent = _longestCommonPrefix(lines)
        if len(indent) == 0:
            indent = " " * 4

        doc += "\n"
        for k, v in obj.__dict__.items():
            if k in fields:
                doc += f"{indent}**{k}**\n\n{indent}.. code-block:: yaml\n\n"
                if isinstance(v, dict):
                    if isinstance(v, Container):
                        v = v.to_dict()
                    lines = yaml.dump(
                        v,
                        sort_keys=False,
                        default_flow_style=False,
                    ).split("\n")
                    indented_lines = "".join(
                        [f"{indent}{indent}{line}\n" for line in lines]
                    )
                    doc += indented_lines
        obj.__doc__ = doc
        return obj

    return _document_attributes
