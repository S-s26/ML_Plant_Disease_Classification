"""Unit tests for class parsing, leaf-ID resolution, and split label mapping."""

from src.data.prepare_splits import parse_class_folder, resolve_leaf_id


def test_parse_healthy_class() -> None:
    crop, cond, bin_id, bin_lbl = parse_class_folder("Tomato___healthy")
    assert crop == "Tomato"
    assert cond == "healthy"
    assert bin_id == 0
    assert bin_lbl == "healthy"


def test_parse_diseased_class() -> None:
    crop, cond, bin_id, bin_lbl = parse_class_folder("Apple___Apple_scab")
    assert crop == "Apple"
    assert cond == "Apple_scab"
    assert bin_id == 1
    assert bin_lbl == "diseased"


def test_resolve_leaf_id_copy_variant() -> None:
    leaf_map = {"rs_early.b 7557": ["Potato___Early_blight:::115.0"]}
    lid1 = resolve_leaf_id("uuid1___RS_Early.B 7557.JPG", "Potato___Early_blight", leaf_map)
    lid2 = resolve_leaf_id("uuid2___RS_Early.B 7557 copy 2.JPG", "Potato___Early_blight", leaf_map)
    assert lid1 == lid2 == "Potato___Early_blight:::115.0"

