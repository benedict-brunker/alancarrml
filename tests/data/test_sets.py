# test_sets.py 

# toy dataset for running tests 
from sklearn.datasets import load_iris 
import pandas as pd 
import numpy as np 
import pytest

test_df = load_iris(as_frame=True).frame

def test_pop_target(): 
    """
    Unit tests for sets.pop() 
    """ 
    from alancarrml.data.sets import pop_target 

    # == Valid Inputs == 

    # Use function with valid inputs   
    X, y = pop_target(test_df, 'target')  
    # check that X is a dataframe 
    assert isinstance(X, pd.DataFrame) 
    # check that Y is a Series 
    assert isinstance(y, pd.Series) 
    # 'target' should not be a column in X 
    assert 'target' not in X.columns 
    # check that the name of y is 'target' 
    assert y.name == 'target' 
    # check that X has four columns  
    assert len(X.columns) == 4 
    # check that X and y have same number of rows 
    assert len(X.index) == len(y.index) 

    # == Invalid Inputs == 
    
    # pass a series to the X parameter 
    X_series = pd.Series(np.arange(1, 10, 1))  
    y = pd.Series(np.arange(2, 20, 2)) 
    df_dict = test_df.to_dict() 
    df_np = test_df.to_numpy() 

    y_df = pd.DataFrame({
        1: np.arange(1, 10, 1), 
        2: np.arange(1, 10, 1), 
        3: np.arange(1, 10, 1), 
    })
    
    # -- ValueErrors -- 

    # Function rejects a series for df parameter 
    with pytest.raises(ValueError):  
        pop_target(X_series, 'target')  

    # Function rejects a dict for df parameter 
    with pytest.raises(ValueError): 
        pop_target(df_dict, 'target')   

    # Rejects numpy input 
    with pytest.raises(ValueError): 
        pop_target(df_np, 'target') 

    # Function rejects non-string parameter for target_col 
    with pytest.raises(ValueError): 
        pop_target(test_df, 35) 
    
    # Function rejects target_col not referencing column name in df 
    with pytest.raises(ValueError): 
        pop_target(test_df, 'flower_type')

    

