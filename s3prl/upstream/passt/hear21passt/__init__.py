__version__ = "0.0.18"


def embeding_size(hop=50, embeding_size=1000):
    embedings = 20 * 60 * (1000 / hop)
    return embedings * embeding_size * 4 / (1024 * 1024 * 1024)  # float32 in GB
