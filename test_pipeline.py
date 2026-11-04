import pytest

def test_bayes_threshold():
    c_fn = 25000.0
    c_fp = 500.0
    threshold = c_fp / (c_fp + c_fn)
    assert pytest.approx(threshold, rel=1e-3) == 0.0196

def test_tier_mapping():
    def get_tier(d):
        if d < 0.15: 
            return 0
        elif d < 0.45: 
            return 1
        else: 
            return 2

    assert get_tier(0.04) == 0
    assert get_tier(0.25) == 1
    assert get_tier(0.70) == 2