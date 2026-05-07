from src.benchmark import load_models

def test_main_excludes_appendix():
    assert 'attention_baseline' not in load_models('main')

def test_appendix_contains_old():
    assert 'attention_baseline' in load_models('appendix')
