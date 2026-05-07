import numpy as np
from models.gradient_boosting import GradientBoostingBaseline

def test_fit_predict():
    X=np.random.randn(20,4); y=np.random.randn(20)
    m=GradientBoostingBaseline().fit(X,y)
    p=m.predict(X[:3])
    assert len(p)==3
