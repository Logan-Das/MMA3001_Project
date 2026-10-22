"""Tests for porkqc.grouping: frame parsing, grouping, grouped split and dHash."""

import numpy as np
import pytest

from porkqc.grouping import (
    FrameId,
    correlation,
    cross_split_near_duplicates,
    deduplicate_by_content,
    deduplicate_frames,
    dhash,
    frame_gaps,
    gap_percentiles,
    group_by_frame_gap,
    grouped_split,
    groups_spanning_splits,
    hamming,
    parse_frame_id,
)


# --- parse_frame_id -------------------------------------------------------

def test_parses_real_roboflow_names():
    # File names taken from the baseline test-split error analysis.
    f = parse_frame_id("frame_with_red_3371_jpg.rf.55a1977413f15016b60c76339853127c.jpg")
    assert f == FrameId("frame_with_red", 3371)


def test_parses_plain_names_and_paths():
    assert parse_frame_id("/x/y/frame_with_red_121.jpg") == FrameId("frame_with_red", 121)


def test_unparseable_name_raises():
    with pytest.raises(ValueError):
        parse_frame_id("IMG_final.jpg")


# --- gaps and grouping ----------------------------------------------------

def test_frame_gaps_ignore_order_and_duplicates():
    np.testing.assert_array_equal(frame_gaps([10, 1, 2, 2, 4]), [1, 2, 6])
    assert gap_percentiles([5]) == {}


def test_group_by_frame_gap_splits_on_large_gaps():
    ids = [FrameId("v", f) for f in [1, 2, 3, 20, 21, 50]]
    g = group_by_frame_gap(ids, max_gap=5)
    assert g[0] == g[1] == g[2]
    assert g[3] == g[4] != g[0]
    assert g[5] not in (g[0], g[3])


def test_same_frame_always_same_group_and_stems_separate():
    ids = [FrameId("a", 7), FrameId("a", 7), FrameId("b", 7)]
    g = group_by_frame_gap(ids, max_gap=0)
    assert g[0] == g[1] and g[2] != g[0]


def test_negative_gap_rejected():
    with pytest.raises(ValueError):
        group_by_frame_gap([FrameId("a", 1)], -1)


def test_deduplicate_keeps_one_copy_preferring_low_priority():
    ids = [FrameId("v", 1), FrameId("v", 1), FrameId("v", 2), FrameId("v", 1)]
    # Index 3 is the un-augmented (priority 0) copy of frame 1.
    assert deduplicate_frames(ids, [1, 1, 1, 0]) == [2, 3]
    # Equal priority: the first copy is kept.
    assert deduplicate_frames(ids, [1, 1, 1, 1]) == [0, 2]
    with pytest.raises(ValueError):
        deduplicate_frames(ids, [0])


def test_deduplicate_by_content_keeps_distinct_pictures():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(100), rng.standard_normal(100)
    ids = [FrameId("v", 1)] * 5 + [FrameId("v", 2)]
    # Frame 1: three copies of picture a (one un-augmented), two copies of a
    # different picture b. Frame 2: one picture.
    feats = [a + 0.05 * rng.standard_normal(100), a, a * 1.0 + 0.02, b, b + 0.05, a]
    prio = [1, 0, 1, 1, 1, 1]
    keep = deduplicate_by_content(ids, prio, feats, 0.9)
    assert keep == [1, 3, 5]          # un-augmented copy of a, first copy of b, frame 2


def test_deduplicate_by_content_validation():
    with pytest.raises(ValueError):
        deduplicate_by_content([FrameId("v", 1)], [0], [])
    with pytest.raises(ValueError):
        deduplicate_by_content([FrameId("v", 1)], [0], [np.ones(3)], min_corr=1.5)
    with pytest.raises(ValueError):
        correlation(np.ones(3), np.ones(4))


def test_groups_spanning_splits_detects_leakage():
    groups = ["a", "a", "b", "c", "c"]
    assert groups_spanning_splits(groups, [0, 2, 0, 1, 1]) == ["a"]
    assert groups_spanning_splits(groups, [0, 0, 1, 2, 2]) == []


# --- grouped_split ----------------------------------------------------------

def make_profiles(n=60, seed=1):
    rng = np.random.default_rng(seed)
    prof = {}
    for i in range(n):
        imgs = rng.integers(1, 8)
        good = rng.integers(0, 2)
        boxes = rng.poisson([0.1, 0.5, 0.1, 1.5, 0.05]) * (imgs - good)
        prof[f"g{i}"] = np.r_[imgs, good, boxes]
    return prof


def test_grouped_split_assigns_every_group_once_and_is_reproducible():
    prof = make_profiles()
    a1 = grouped_split(prof, seed=3)
    a2 = grouped_split(prof, seed=3)
    assert a1 == a2
    assert set(a1) == set(prof)
    assert set(a1.values()) <= {0, 1, 2}


def test_grouped_split_roughly_matches_ratios():
    prof = make_profiles(200)
    a = grouped_split(prof, (0.7, 0.15, 0.15))
    imgs = np.zeros(3)
    for g, s in a.items():
        imgs[s] += prof[g][0]
    frac = imgs / imgs.sum()
    np.testing.assert_allclose(frac, [0.7, 0.15, 0.15], atol=0.05)


def test_rare_class_reaches_test_split():
    # Only two groups contain class 2; one should go to a non-train split.
    prof = {f"g{i}": np.array([3, 0, 0, 0, 0, 3, 0]) for i in range(20)}
    prof["rare1"] = np.array([2, 0, 0, 0, 2, 0, 0])
    prof["rare2"] = np.array([2, 0, 0, 0, 2, 0, 0])
    a = grouped_split(prof)
    assert {a["rare1"], a["rare2"]} != {0}


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)])
def test_grouped_split_rejects_bad_ratios(ratios):
    with pytest.raises(ValueError):
        grouped_split({"g": np.array([1, 0])}, ratios)


# --- dHash near-duplicate check ---------------------------------------------

def test_dhash_identical_and_exposure_shift():
    rng = np.random.default_rng(0)
    img = rng.random((54, 72)) * 255
    h = dhash(img)
    assert h.shape == (64,)
    assert hamming(h, dhash(img)) == 0
    # A uniform exposure change (like Roboflow's augmentation) keeps the hash.
    assert hamming(h, dhash(img * 0.8 + 10)) == 0


def test_dhash_different_images_differ():
    rng = np.random.default_rng(0)
    a, b = rng.random((54, 72)), rng.random((54, 72))
    assert hamming(dhash(a), dhash(b)) > 15


def test_dhash_input_validation():
    with pytest.raises(ValueError):
        dhash(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        dhash(np.zeros((4, 4)))


def test_cross_split_near_duplicates():
    rng = np.random.default_rng(0)
    train = [rng.random((40, 50)) for _ in range(5)]
    test = [train[2] + 0.01, rng.random((40, 50))]
    th = np.array([dhash(x) for x in train])
    qh = np.array([dhash(x) for x in test])
    np.testing.assert_array_equal(cross_split_near_duplicates(qh, th), [True, False])
    assert cross_split_near_duplicates(qh, np.zeros((0, 64), bool)).tolist() == [False, False]
