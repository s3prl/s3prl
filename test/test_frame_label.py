from s3prl.dataio.dataset.frame_label import (
    chunk_labels_to_frame_tensor_label,
    chunking,
)


def test_chunking():
    chunks = list(chunking(0.0, 8.5, 2.0, 1.0, False))
    assert len(chunks) == 7

    chunks = list(chunking(1.1, 8.5, 2.0, 1.0, True))
    assert len(chunks) == 8


def test_frame_tensor_label():
    labels = [
        (0, 3.0, 4.1),
        (1, 1.2, 3.2),
    ]
    label = chunk_labels_to_frame_tensor_label(1.5, 4.0, labels, 3, 160)
    assert label[-1, 0] == 1
    assert label[0, 1] == 1
