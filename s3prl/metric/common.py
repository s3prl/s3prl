def accuracy(xs, ys, item_same_fn=None):
    if isinstance(xs, (tuple, list)):
        assert isinstance(ys, (tuple, list))
        return _accuracy_impl(xs, ys, item_same_fn)
    elif isinstance(xs, dict):
        assert isinstance(ys, dict)
        keys = sorted(list(xs.keys()))
        xs = [xs[k] for k in keys]
        ys = [ys[k] for k in keys]
        return _accuracy_impl(xs, ys, item_same_fn)
    else:
        raise ValueError


def _accuracy_impl(xs, ys, item_same_fn=None):
    item_same_fn = item_same_fn or (lambda x, y: x == y)
    same = [int(item_same_fn(x, y)) for x, y in zip(xs, ys)]
    return sum(same) / len(same)
