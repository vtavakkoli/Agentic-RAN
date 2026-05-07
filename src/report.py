from pathlib import Path
import pandas as pd

def main():
    out=Path('results'); out.mkdir(exist_ok=True)
    main_df=pd.read_csv(out/'main_benchmark.csv') if (out/'main_benchmark.csv').exists() else pd.DataFrame()
    app_df=pd.read_csv(out/'appendix_benchmark.csv') if (out/'appendix_benchmark.csv').exists() else pd.DataFrame()
    html='''<html><body><h1>Executive summary</h1>
<h2>Main benchmark models</h2><h2>Forecasting leaderboard</h2><h2>Control / agentic policy leaderboard</h2><h2>Proposed SafeGraphAgent-RAN</h2><h2>Recommended deployment candidate</h2><h2>Appendix / extended temporal baselines</h2><h2>Scientific wording note</h2><h2>Dataset/source summary</h2><h2>Prediction plots</h2>
<p>The main benchmark focuses on time-aware tabular, residual, graph-agentic, and safe-control models. Older temporal baselines are moved to the appendix because they underperform in the current dataset and should not drive the main scientific claim.</p>
<p>Pseudo-label action metrics are not sufficient to prove real control quality. Control quality is assessed through offline reward, slice-specific KPIs, action-switch behavior, safe fallback rate, and safety-constraint behavior.</p>
'''+main_df.to_html(index=False)+app_df.to_html(index=False)+'''</body></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')
if __name__=='__main__': main()
