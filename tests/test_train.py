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

def test_preprocess_origin_cols():
    actual = pd.DataFrame({'Origin': [1]})
    expected = pd.DataFrame({'Origin': ['India', 'USA','Germany']})    
    assert all(preprocess_origin_cols(actual) ==  expected)

# def test_preprocess_origin_cols():
#     actual = pd.DataFrame({'Origin': [1, 2,3]})
#     expected = pd.DataFrame({'Origin': ['India', 'USA','Germany']})    
#     assert all(preprocess_origin_cols(actual) ==  expected)

# def test_preprocess_origin_cols_no_of_unique_values():
#     actual = pd.DataFrame({'Origin': [1,2,3,4]})
#     actual_len = actual['Origin'].nunique()
#     expected = preprocess_origin_cols(actual)
#     expected_len = acc['Origin'].nunique()
#     # with pytest.raises(AssertionError):
#     assert actual_len == expected_len


    # expected = pd.DataFrame({'Origin': ['India', 'USA','Germany','five']})
    # with pytest.raises(ValueError):


# @pytest.mark.parametrize("test_size",[0.2,-0.2,0.5,-0.5])
# def test_strat_split(test_size):
#     assert test_size <=0.3 and test_size >=0.1



    # preprocessed_df['Origin'].nunique()    
    
    