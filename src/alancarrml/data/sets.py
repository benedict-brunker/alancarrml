# top-level dependencies 
import pandas as pd 

# pop_target 
def pop_target(df, target_col): 
    """ 
    Extracts the target variable from the input dataframe. 
    Args: 
        - df (pd.DataFrame): DataFrame from which to pop out the target variable. 
        - target_col (str): Name of target variable in df. 
    Returns: 
        - X (pd.DataFrame): a featureset 
        - y (pd.Series): target variable 
    """ 
    # validate input types 
    if not isinstance(df, pd.DataFrame): 
        raise ValueError("Parameter df must be of type pd.DataFrame.")
    if not isinstance(target_col, str): 
        raise ValueError("Parameter target_col must be a string") 
    if target_col not in df.columns: 
        raise ValueError(f"target_col {target_col} does not reference any column in dataframe {df}")
    
    # extract target column y 
    y = df[target_col] 
    # drop y from dataframe 
    X = df.drop(target_col, axis=1) 
    # return X and y 
    return X, y 