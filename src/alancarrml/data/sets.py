# top-level dependencies 
import pandas as pd  
import numpy as np 
from scipy import sparse
import os  
import json 

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
        raise TypeError("Parameter df must be of type pd.DataFrame.")
    if not isinstance(target_col, str): 
        raise TypeError("Parameter target_col must be a string") 
    if target_col not in df.columns: 
        raise KeyError(f"target_col {target_col} does not reference any column in dataframe {df}")
    
    # extract target column y 
    y = df[target_col] 
    # drop y from dataframe 
    X = df.drop(target_col, axis=1) 
    # return X and y 
    return X, y 


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, save_dir=None, filenames=[], save_as='numpy'): 
    """
    Saves datasets to path with numpy if they exist. 

    Parameters 
    _____________________________________________________________
    X_train: array or matrix 
        Features for the training set. 
    y_train: array, matrix or iterable 
        Target for the training set. 
    X_val: array or matrix 
        Features for the validation set. 
    y_val: array, matrix or iterable 
        Target for the validation set. 
    X_test: array or matrix  
        Features for the test set. 
    y_test: array, matrix or iterable 
        Target for the test set. 
    path: str 
        Path to save directory. Will default to '../data/processed/'   
    filenames: list 
        Provide a list of custom filenames, as strings, in order of arguments given. 
        Otherwise defaults to "X_train.ext", "y_train.ext" etc. 
        E.g. ["X_train_2.npy", "y_train_2.npy", ..., "y_test_2.npy"]
    
    Returns 
    _____________________________________________________________
    None 
    """ 

    # default save path if none given  
    if save_dir is None: 
        save_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed') 
    
    # Acceptable types dict {'object': 'ext'} 
    acceptable_types = {
        'ndarray': '.npy', 
        'sparse': '.npz', 
        'DataFrame': '.csv', 
        'series': '.csv', 
        'list': '.npy', 
        'dict': '.json'
    }

    i = 0 # counter for tracking position in the filenames list 

    if X_train is not None: 
        i = save_set(X_train, "x_train", acceptable_types, i, filenames, save_dir) 
    
    if y_train is not None: 
        i = save_set(y_train, "y_train", acceptable_types, i, filenames, save_dir) 

    if X_val is not None: 
        i = save_set(X_val, "x_val", acceptable_types, i, filenames, save_dir) 
    
    if y_val is not None: 
        i = save_set(y_val, "y_val", acceptable_types, i, filenames, save_dir) 
    
    if X_test is not None: 
        i = save_set(X_test, "x_test", acceptable_types, i, filenames, save_dir) 
    
    if y_test is not None: 
        i = save_set(y_test, "y_test", acceptable_types, i, filenames, save_dir) 
    
    return 

# Helper function that wraps class and filename inference, and saves 
def save_set(dataset, pattern, acceptable_types, increment, filenames, save_dir): 
    # infer type 
    dataset_type = infer_class(dataset) 
    if isinstance(dataset_type, str):  
        if filenames:
            dataset_filename = infer_filename(filenames, pattern, increment)
            # update the increment 
            increment += 1 
        else: 
            dataset_filename = pattern + acceptable_types.get(dataset_type) 
        
        save_type(dataset, dataset_type, save_dir, dataset_filename)
    # return the increment 
    return increment 


# Helper function to infer object type and whether allowable 
def infer_class(dataset): 
    acceptable_types = ['ndarray', 'sparse', 'DataFrame', 'series', 'list', 'dict']
    set_type = type(dataset) 
    # split at ' and take object 1 
    clean_set_type = set_type.split('\'')[1] 
    # split at . 
    classes = clean_set_type.split(".") 
    # check if object type in list of acceptable_types 
    for a in acceptable_types: 
        if a in classes: 
            return a 
    else:
        # datatype not acceptable, return False 
        raise ValueError(f""" 
            Object of type {set_type} not permissible. 
            Acceptable object types are: 
                - np.ndarray
                - scipy.sparse matrix 
                - pd.DataFrame 
                - pd.Series 
                - list 
                - dict 
                         """)  

# Helper function that saves depending on object type 
def save_type(dataset, object_type, path, filename): 

    if filename: 
        path = os.path.join(path, filename) 

    if object_type == 'ndarray': 
        np.save(path, dataset) 
        return True if os.path.isfile(path) else False

    elif object_type == 'sparse': 
        sparse.save_npz(path, dataset)  
        return True if os.path.isfile(path) else False

    elif object_type in ['DataFrame', 'series']: 
        # add .csv extension to path if not there already 
        if path.split(".")[-1] != 'csv': 
            path = path + ".csv" 
        dataset.to_csv(path)  
        return True if os.path.isfile(path) else False

    elif object_type == 'list': 
        dataset_np = np.array(dataset) 
        np.save(path, dataset) 
        return True if os.path.isfile(path) else False 
    
    elif object_type == 'dict': 
        # add .json extension if needed 
        if path.split(".")[-1] != 'json': 
            path = path + ".json" 
        with open(path, 'w') as j:
            json.dump(dataset, j) 
        return True if os.path.isfile(path) else False
    
# Helper to infer correct filename 
def infer_filename(filenames, pattern, i): 

    for f in filenames: 
        if f.lower().split().contains(pattern): 
            return f 
    # if none found, select the first object and alert the user 
    else: 
        print(f"""
                Filename for X_train could not be inferred, defaulted to {filenames[i]}. 
                To save under a different name, run this function again and pass a name similar to 'x_train' to the filenames argument.
            """) 
        return filenames[i]