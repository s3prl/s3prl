from .expert import UpstreamExpert as _UpstreamExpert


def passt_base(**kwds):
    return _UpstreamExpert("base", **kwds)


def passt_base2level(**kwds):
    return _UpstreamExpert("base2level", **kwds)


def passt_base2levelmel(**kwds):
    return _UpstreamExpert("base2levelmel", **kwds)


def passt_base20sec(**kwds):
    return _UpstreamExpert("base20sec", **kwds)


def passt_base30sec(**kwds):
    return _UpstreamExpert("base30sec", **kwds)


def passt_hop100base(**kwds):
    return _UpstreamExpert("hop100base", **kwds)


def passt_hop100base2lvl(**kwds):
    return _UpstreamExpert("hop100base2lvl", **kwds)


def passt_hop100base2lvlmel(**kwds):
    return _UpstreamExpert("hop100base2lvlmel", **kwds)


def passt_hop160base(**kwds):
    return _UpstreamExpert("hop160base", **kwds)


def passt_hop160base2lvl(**kwds):
    return _UpstreamExpert("hop160base2lvl", **kwds)


def passt_hop160base2lvlmel(**kwds):
    return _UpstreamExpert("hop160base2lvlmel", **kwds)


# FIXME: url seems corrupted
# def passt_openmic2008(**kwds):
#     return _UpstreamExpert("openmic2008", **kwds)
