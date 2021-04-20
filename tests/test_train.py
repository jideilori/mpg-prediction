from scripts.train import load_data,get_feat_labels
from scripts.train import strat_split

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

# @pytest.mark.parametrize("test_size",[0.2,-0.2,0.5,-0.5])
# def test_strat_split(test_size):
#     assert test_size <=0.3 and test_size >=0.1



    # preprocessed_df['Origin'].nunique()    
    
    