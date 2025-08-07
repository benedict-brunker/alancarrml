# test_sets.py 

# toy dataset for running tests  
import pandas as pd 
import numpy as np 
import pytest 

from sklearn.datasets import load_iris 

from alancarrml.data.sets import pop_target 

test_df = load_iris(as_frame=True).frame

    # === Solution Unit Tests ===  # 

# creating dummy data with pytest fixtures 
@pytest.fixture
def features_fixture(): 
    features_data = [
        [1, 25, "Junior"], 
        [2, 33, "Confirmed"], 
        [3, 42, "Manager"] 
    ] 
    return pd.DataFrame(features_data, columns=["employee_id", "age", "level"]) 

# the @pytest.fixture decorator associates some value with the value returned by the function below the decorator 
# e.g. in later tests below, target_fixture is always associated with the return value of the target_fixture() function 
@pytest.fixture 
def target_fixture(): 
    target_data = [5, 10, 20] 
    return pd.Series(target_data, name="salary", copy=False) 

def test_pop_target_with_data_fixture(features_fixture, target_fixture): 
    input_df = features_fixture.copy() 
    input_df["salary"] = target_fixture 

    features, target = pop_target(df=input_df, target_col='salary') 

    pd.testing.assert_frame_equal(features, features_fixture) 
    pd.testing.assert_series_equal(target, target_fixture) 

def test_pop_target_no_col_found(features_fixture, target_fixture): 
    input_df = features_fixture.copy() 

    with pytest.raises(TypeError): 
        features, target = pop_target(df=input_df, target_col=None) 

def test_pop_target_df_none(features_fixture, target_fixture): 
    input_df = features_fixture.copy() 

    with pytest.raises(TypeError): 
        features, target = pop_target(df=None, target_col='salary') 


    # === Original attempt at unit tests ===  # 

def test_pop_target(): 
    """
    Unit tests for sets.pop() 
    """ 
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
    with pytest.raises(TypeError):  
        pop_target(X_series, 'target')  

    # Function rejects a dict for df parameter 
    with pytest.raises(TypeError): 
        pop_target(df_dict, 'target')   

    # Rejects numpy input 
    with pytest.raises(TypeError): 
        pop_target(df_np, 'target') 

    # Function rejects non-string parameter for target_col 
    with pytest.raises(TypeError): 
        pop_target(test_df, 35) 
    
    # Function rejects target_col not referencing column name in df 
    with pytest.raises(KeyError): 
        pop_target(test_df, 'flower_type')

    

