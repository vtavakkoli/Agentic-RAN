from src.report import main
from pathlib import Path

def test_sections_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('results').mkdir()
    Path('results/main_benchmark.csv').write_text('model,r2\na,0.1\n')
    main()
    html=Path('results/report.html').read_text()
    assert 'Executive summary' in html and 'Control / agentic policy leaderboard' in html
