"""Tests for porkqc.labels: YOLO label reading and input validation."""

import numpy as np
import pytest

from porkqc.labels import LabelFormatError, class_counts, label_path_for, read_yolo_label


def write(tmp_path, text, name="a.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_reads_valid_boxes(tmp_path):
    p = write(tmp_path, "3 0.5 0.5 0.2 0.1\n0 0.1 0.9 0.05 0.05\n")
    arr = read_yolo_label(p)
    assert arr.shape == (2, 5)
    assert arr[0, 0] == 3 and arr[1, 0] == 0
    np.testing.assert_allclose(arr[0, 1:], [0.5, 0.5, 0.2, 0.1])


def test_missing_and_empty_files_mean_good_tray(tmp_path):
    assert read_yolo_label(tmp_path / "missing.txt").shape == (0, 5)
    assert read_yolo_label(write(tmp_path, "\n\n")).shape == (0, 5)


@pytest.mark.parametrize(
    "line",
    [
        "7 0.5 0.5 0.1 0.1",          # class id out of range
        "1.5 0.5 0.5 0.1 0.1",        # non-integer class id
        "1 1.2 0.5 0.1 0.1",          # coordinate outside [0, 1]
        "1 0.5 0.5 0.0 0.1",          # zero width
        "1 0.5 0.5 0.1",              # too few values
        "1 0.1 0.1 0.2 0.1 0.3 0.3",  # polygon export, not boxes
        "1 a 0.5 0.1 0.1",            # non-numeric
    ],
)
def test_rejects_malformed_lines(tmp_path, line):
    with pytest.raises(LabelFormatError):
        read_yolo_label(write(tmp_path, line))


def test_error_message_names_file_and_line(tmp_path):
    p = write(tmp_path, "0 0.5 0.5 0.1 0.1\n9 0.5 0.5 0.1 0.1\n", "tray.txt")
    with pytest.raises(LabelFormatError, match="tray.txt:2"):
        read_yolo_label(p)


def test_label_path_for():
    p = label_path_for("/data/test/images/frame_1.jpg")
    assert p.as_posix() == "/data/test/labels/frame_1.txt"
    with pytest.raises(ValueError):
        label_path_for("/data/test/frame_1.jpg")


def test_class_counts(tmp_path):
    a = write(tmp_path, "3 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n", "a.txt")
    b = write(tmp_path, "1 0.5 0.5 0.1 0.1\n", "b.txt")
    c = write(tmp_path, "", "c.txt")
    out = class_counts([a, b, c])
    assert out["boxes"] == {3: 2, 1: 1}
    assert out["images"] == {3: 1, 1: 1}
    assert out["background"] == 1 and out["total"] == 3
