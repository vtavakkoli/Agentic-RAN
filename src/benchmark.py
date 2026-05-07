from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.ranking import rank_forecasting, rank_control


def load_models(scope: str):
    cfg = json.loads(Path('configs/benchmark_models.yaml').read_text())
    if scope=='main': return cfg['main_models']
    if scope=='appendix': return cfg['appendix_models']
    if scope=='foundation': return cfg['optional_foundation_models']
    if scope=='all': return cfg['main_models']+cfg['appendix_models']
    raise ValueError(scope)

def mock_metrics(model):
    return {"model":model,"r2":0.7,"rmse":0.3,"mae":0.2,"smape":10.0,"wmape":9.0,"composite_score":60.0,"peak_mae":0.3,"peak_rmse":0.4,"peak_r2":0.6,
            "average_reward":0.1 if 'graph' in model or 'agentic' in model or 'safe' in model else None,
            "cumulative_reward":10.0 if 'graph' in model or 'agentic' in model or 'safe' in model else None,
            "action_switch_rate":0.2 if 'graph' in model or 'agentic' in model or 'safe' in model else None,
            "safe_fallback_rate":0.05 if 'graph' in model or 'agentic' in model or 'safe' in model else None,
            "avg_decision_confidence":0.75 if 'graph' in model or 'agentic' in model or 'safe' in model else None,
            "pseudo_label_action_accuracy":0.5 if 'graph' in model or 'agentic' in model or 'safe' in model else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark-scope',default='main',choices=['main','appendix','all','foundation']); args=ap.parse_args()
    models=load_models(args.benchmark_scope)
    rows=[mock_metrics(m) for m in models]
    df=pd.DataFrame(rows)
    out=Path('results'); out.mkdir(exist_ok=True)
    if args.benchmark_scope in ('main','all'): df[df['model'].isin(load_models('main'))].to_csv(out/'main_benchmark.csv',index=False)
    if args.benchmark_scope in ('appendix','all'): df[df['model'].isin(load_models('appendix'))].to_csv(out/'appendix_benchmark.csv',index=False)
    rank_forecasting(df).to_csv(out/'model_ranking.csv',index=False)
    rank_control(df.dropna(subset=['average_reward'], how='all')).to_csv(out/'control_ranking.csv',index=False)
    df[df['model']=='safegraphagent_ran'].to_csv(out/'safegraphagent_ran_metrics.csv',index=False)

if __name__=='__main__': main()
