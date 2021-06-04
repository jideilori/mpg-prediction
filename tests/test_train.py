from scripts.train import load_data,get_feat_labels,preprocess_origin_cols
from scripts.train import strat_split
import pandas as pd
import pytest

data =  load_data('./data/raw/auto-mpg.data')
def test_is_number_of_columns_8():
    no_of_cols = len([ i for i in data.columns])
    assert no_of_cols == 8

def test_strat_split():
    with pytest.raises(TypeError):
        assert strat_split(data,8,0.2,42)

def test_get_feat_labels():
    assert get_feat_labels(data, 'MPG')
