from falak3d.presets import Preset
from falak3d.prompting import clean, glossary_translate, looks_english, to_english


def test_english_is_left_alone():
    assert to_english("a red wooden chair") == "a red wooden chair"


def test_french_without_llm_uses_glossary():
    assert not looks_english("une chaise en bois rouge")
    assert glossary_translate("une chaise en bois rouge") == "wooden red chair"


def test_unreachable_llm_falls_back_to_glossary():
    out = to_english("une lampe", ollama_url="http://127.0.0.1:9", timeout=0.5)
    assert out == "lamp"


def test_clean_collapses_whitespace_and_truncates():
    assert clean("  a\n\nmug  ") == "a mug"
    assert len(clean("x" * 500)) == 200


def test_preset_template():
    preset = Preset(name="jeu", title="Jeu", template="low poly {subject}, game asset")
    assert preset.prompt(" barrel ") == "low poly barrel, game asset"
