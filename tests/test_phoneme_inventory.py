import pytest
from neurobridge.data_pipeline import PhonemeInventory

def test_phoneme_inventory_initialization():
    symbols = ["a", "b", "c"]
    inventory = PhonemeInventory(symbols)
    # 'sil' should be added automatically if not present
    assert "sil" in inventory.id_to_symbol
    assert inventory.num_classes == 4

def test_phoneme_inventory_encoding():
    symbols = ["a", "b", "c"]
    inventory = PhonemeInventory(symbols)
    assert inventory.encode("a") == inventory.symbol_to_id["a"]
    assert inventory.encode("nonexistent") == inventory.symbol_to_id["sil"]
    assert inventory.encode("  A  ") == inventory.symbol_to_id["a"]

def test_phoneme_inventory_decoding():
    symbols = ["a", "b", "c"]
    inventory = PhonemeInventory(symbols)
    assert inventory.decode(inventory.symbol_to_id["a"]) == "a"
    assert inventory.decode(999) == "c" # Clamps to max index
    assert inventory.decode(-1) == "sil" # Clamps to min index
