# === TOP-LEVEL DEPENDENCIES ===  
import pandas as pd  
import numpy as np 
from scipy import sparse
import os  
import json 

# === CORE FUNCTIONS === 

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

# save_sets
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

# load_sets
def load_sets(load_dir=None): 
    """
    Loads locally saved sets if they exist. 

    Parameters 
    __________________________________________________________________________________________________
    load_dir: string | pathlike 
        Path to directory from which to load sets. If None will default to '../data/processed/' 
    
    Returns 
    __________________________________________________________________________________________________
    datasets: dict 
        All dataset objects in load_dir, packaged as a dict for consistency, formatted as {'filename': dataset} 
    """
    # construct default path if None given 
    if load_dir is None: 
        load_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed') 
    
    while True: 
        if not os.path.exists(load_dir): 
            load_dir = input(f"""
                             Path {load_dir} could not be found. 
                             Please input path to directory in which data sets are stored.
                             """)
        else:
            break 
    
    # list and count contents of directory 
    pathnames = os.listdir(load_dir) 
    if not pathnames: 
        raise FileNotFoundError(f"""
              Directory {load_dir} did not contain any files to load.
              """)
    
    # check that at least one file exists in load_dir 
    datasets = {} 
    n_files = 0 # counts number of valid files
    valid_extensions = ['csv', 'npy', 'npz', 'json']
    for p in pathnames: 
        extension = p.split(".")[-1]
        if (os.path.isfile(p)) and (extension in valid_extensions):
            full_path = os.path.join(load_dir, p) 
            datasets[p] = load_set(full_path, extension) 
            n_files += 1 

    if n_files < 1: 
        raise FileNotFoundError(f"""
                Directory {load_dir} did not contain any files to load.
                                """)
    
    if datasets:
        return datasets 
    else:
        raise FileNotFoundError(f"""
                                No files were successfully loaded from {load_dir} 
                                """)


# subset
def subset(start_index: int, end_index: int, features=None, target=None, df=None): 
    """
    Subsets features and target based on start_index and end_index. 

    Parameters 
    __________________________________________________________________________________________________________________________________________________________________________

    start_index: int 
        Index position from which to subset both target and features, inclusive. 
    end_index: int 
        Index position to which to subset both target and features, exclusive.
    target: pd.DataFrame, default None. 
        A DataFrame containing features. If None, df cannot also be None. 
    features: pd.DataFrame 
        A DataFrame containing the target variable(s). If None, df cannot also be None.   
    df: pd.DataFrame 
        DataFrame to subset. 
        If a DataFrame is passed to this argument, the function returns the subsetted DataFrame rather than target and features separately. 
        If a DataFrame is passed to this argument, both features and target should be left as None.

    Returns
    __________________________________________________________________________________________________________________________________________________________________________
    features_subset: pd.DataFrame 
        Subset of features from start_index (inclusive) to end_index (exclusive) 
    target_subset: pd.DataFrame 
        Subset of target from start_index (inclusive) to end_index (exclusive). 

    OR 

    df_subset: pd.DataFrame 
        Subset of entire df from start_index (inclusive) to end_index (exclusive) 
    """
    # check input validity 
    if features: 
        if not isinstance(features, (pd.DataFrame, pd.Series)): 
            raise TypeError(f"Expected pd.DataFrame object for features parameter, got {type(features)}")
        if df:
            raise ValueError("Values can be passed to features or df parameters, not both.") 
        if not target: 
            raise ValueError(f"""
                             If a pd object is passed to the features parameter, a similar object must also be passed to the target parameter, not {type(target)}
                             """)

    if target: 
        if not isinstance(target, (pd.DataFrame, pd.Series)): 
            raise TypeError(f"Expected pd.DataFrame or pd.Series object for target parameter, got {type(target)}") 
        if df: 
            raise ValueError("Values can be passed to target or df parameters, not both.") 
        if not features:
            raise ValueError(f"""
                             If a pd object is passed to the target parameter, a similar object must also be passed to the features parameter, not {type(features)})
                            """) 
    
    if df: 
        if not isinstance(df, (pd.DataFrame)): 
            raise TypeError(f"Expected pd.DataFrame for df parameter, got {type(df)}") 
        if target or features:
            raise ValueError("If a value is passed to df, values may not be passed to target or features parameters")

    if not isinstance(start_index, int): 
        raise TypeError(f"Expected int for start_index parameter, got {type(start_index)}") 

    if not isinstance(end_index, int): 
        raise TypeError(f"Expected int for end_index paramter, got {type(end_index)}") 
    
    # ensure features and target indices are identical  
    if features and target:
        if not all(features.index == target.index): 
            raise ValueError("Features and target indices are misaligned") 
        else:
        # return subsets (IndexError will be raised automatically on failure) 
            return features.iloc[start_index:end_index].copy(), target.iloc[start_index:end_index].copy()
    elif df: 
        return df.iloc[start_index:end_index].copy()
    else:
        raise ValueError("pd objects must either be passed to both features and target parameter, or to df parameter")

# split_sets_by_time 
def split_sets(df:pd.DataFrame, target_col='', time_col='', format=None, errors='raise', val_ratio=0.2, test_ratio=0.2, by_time=True, pop=True): 
    """
    Converts target_col to datetime and splits df by time values in target_col.

    Parameters 
    ________________________________________________________________________________________________________________________________________________________
    df: pd.DataFrame 
        DataFrame to split. 
    target_col: str 
        Name of column in df containing target variable. 
        If None, '' or other Falsy value is passed, df is returned without popping target from features, equivalent to pop=False.
    time_col: str, default None
        Name of column in df containing values parsable as datetime, by which to split the dataframe. 
        If None, '' or other Falsy value is passed, assumes dataframe is sorted by time and returns a simple split, equivalent to by_time=False. 
    format: str, default none 
        Strftime to parse time.  
    errors: {'ignore', 'raise', 'coerce'}, default 'raise' 
        How to handle errors in datetime parsing. 
        - If 'raise' then invalid parsing will raise an exception. 
        - If 'coerce' then invalid parsing will be set as NaT. 
        - If 'ignore' then invalid parsing will return the input. 
    val_ratio: float 
        Ratio of df to return as validation data. 
        Set to 0 or None to only return a two-way train-test split. 
    test_ratio: float 
        Ratio of df to return as test data. 
        Set to 0 or None to only return a two-way train-val split. 
        If both val_ratio and test_ratio are set to 0 or None, the original df will be returned unsplit. 
        However this behaviour is inappropriate to the function. 
    by_time: bool, default True 
        Whether to split df by datetime values in time_col. 
        Setting this parameter to False performs a simple split by df index ordering as given. 
    pop: bool, default True 
        Whether to return features and target separately (True) or return split dataframes (False) 

    Returns 
    ________________________________________________________________________________________________________________________________________________________
    datasets: dict{'dataset_name': pd.DataFrame|pd.Series} 
        Dict in which keys describe dataset and values are the split datasets. 
        A dict is returned for more adaptable behaviour. 
    """
    # Validate inputs 
    if not isinstance(df, pd.DataFrame): 
        raise TypeError(f"Expected pd.DataFrame for df parameter, got {type(df)}") 
    if not isinstance(target_col, (str, None)): 
        raise TypeError(f"Expected string or None type for target_col parameter, got {type(target_col)}") 
    if not isinstance(time_col, (str, None)): 
        raise TypeError(f"Expected string or None type for time_col parameter, got {type(time_col)}")
    if not isinstance(val_ratio, (float, int, None)):
        raise TypeError(f"Expected int, float or None type for val_ratio, got {type(val_ratio)}")  
    if not isinstance(test_ratio, (float, int, None)):  
        raise TypeError(f"Expected int, float or None type for test_ratio, got {type(test_ratio)}") 
    

    # ensure val_ratio and test_ratio values between 0 and 1 
    if val_ratio < 0 or val_ratio > 1: 
        raise ValueError(f"val_ratio parameter must be a float between 0 and 1, got {type(val_ratio)}")
    if test_ratio < 0 or test_ratio > 1: 
        raise ValueError(f"test_ratio must be a floating-point value between 0 and 1, got {test_ratio}") 
    
    # If a val_ratio is given but no test_ratio, pass val_ratio to test_ratio and alert the user 
    if (val_ratio is not None) and (test_ratio is None): 
        test_ratio = val_ratio 
        val_ratio = None 
        print(f"""
              Valid argument given for val_ratio of {val_ratio} but test_ratio was {test_ratio}. 
              val_ratio will be treated as test_ratio, so return values will look like: 
              \{'X_train': pd.DataFrame, 
                'X_test': pd.DataFrame, 
                etc.    
              }
              """)
    
    
    # check that target_col and time_col are in df if given 
    if target_col and target_col not in df.columns: 
        raise IndexError(f"target_col {target_col} could not be found in df.") 
    if time_col and time_col not in df.columns:
        raise IndexError(f"time_col {time_col} could not be found in df.")
    
    # -- Main logic   -- 

    # try casting target_col to datetime 
    if by_time and time_col:
        try: 
            datetime_col = pd.to_datetime(df[time_col], errors=errors, format=format) 
            df = df.sort_values(time_col, ascending=True)               # sort values in df by time_col
        
        except pd.errors.ParserError as e: 
            print(f"""
                ParserError encountered in parsing {time_col} to datetime. 
                Defaulting to simple index split instead (equivalent to by_time=False) 
                Function must be run again to attempt splitting by datetime values in {time_col}. 
                Error message: 
                    {e} 
                """)
            by_time = False
        except ValueError as e: 
            print(f"""
                ValueError encountered in parsing {time_col} to datetime. 
                Defaulting to simple index split instead (equivalent to by_time=False) 
                Function must be run again to attempt splitting by datetime values in {time_col}. 
                Error message: 
                    {e} 
                """)  
            by_time = False      
    else:      # otherwise we have to assume df is already sorted as desired 
        print("No argument or invalid argument passed to time_col, assuming df is already sorted as desired.") 


    if test_ratio is not None and test_ratio > 0: 
            
            len_idx = len(df.index) 
            test_cutoff = round(len_idx - (len_idx * test_ratio))
            
            if pop:  # check whether we should pop out target first
                X, y = pop_target(df, target_col) 
                X_data, y_data = subset(start_index=0, end_index=test_cutoff, features=X, target=y)
                X_test, y_test = subset(start_index=test_cutoff, end_index=len_idx, features=X, target=y) 
                len_data_idx = len(X_data.index) # potentially needed for next train-val split 

            else: 
                df_data = subset(start_index=0, end_index=test_cutoff, df=df)  
                df_test = subset(start_index=test_cutoff, end_index=len_idx, df=df)   
                len_data_idx = len(df_data.index)   # potentially needed for next train-val split
        
            if val_ratio is not None and val_ratio > 0: 

                val_cutoff = round(len_data_idx - (len_data_idx * test_ratio))

                if pop: 
                    X_train, y_train = subset(start_index=0, end_index=val_cutoff, features=X_data, target=y_data) 
                    X_val, y_val = subset(start_index=val_cutoff, end_index=len_data_idx, features=X_data, target=y_data) 

                    return {
                        'X_train': X_train, 
                        'y_train': y_train, 
                        'X_val': X_val, 
                        'y_val': y_val, 
                        'X_test': X_test, 
                        'y_test': y_test 
                    }
            # if a test_ratio is given but no val_ratio, return two-way data-test split 
            elif pop: 
                return {
                    'X_data': X_data, 
                    'y_data': y_data, 
                    'X_test': X_test, 
                    'y_test': y_test 
                } 
            else: 
                return {
                    'df_data': df_data, 
                    'df_test': df_test 
                }
    
    elif pop:   # if test_ratio is None, alert the user, optionally popping and returning inputs 
        X, y = pop_target(df, target_col) 
        print(f""" 
                test_ratio was {test_ratio} and val_ratio was {val_ratio}. 
                Original df will be split between features and target as: 
                \{
                    'X': features,
                    'y': target
                }
              """)
        return {
            'X': X, 
            'y': y 
        }
    
    else:  # otherwise we just return the input df and alert the user 
        print(f""" 
                test_ratio was {test_ratio} and val_ratio was {val_ratio}. 
                Original df will be returned unchanged as: 
                \{
                    'df': df
                }
              """)
        return {
            'df': df 
        }


# split_sets_random 
def split_sets_random(features, target, test_ratio): 
    """
    Splits features and target three-ways into training, validation and test sets according to test_ratio. 
    Size of validation set will be identical to size of test set. 
    This is therefore different to a two-step split performed by e.g. sklearn's train_test_split function, in which case the data-test split may end up of different size to the train-val split. 

    Parameters
    ____________________________________________________________________________________________________________________________________________________________________________________________________
    features: pd.DataFrame 
        A DataFrame containing features to be split randomly. 
    target: pd.DataFrame | pd.Series 
        A DataFrame containing features to be split in alignment with features. features and target must have aligned indexes. 
    test_ratio: float 
        Ratio of data to reserve for both validation and test sets.     

    Returns
    ____________________________________________________________________________________________________________________________________________________________________________________________________
    Dict of randomly split datasets, keyed by names, e.g.
        {
            'X_train': pd.DataFrame, 
            'X_val': pd.DataFrame, 
            'X_test': pd.DataFrame, 
            'y_train': pd.DataFrame | pd.Series, 
            'y_val': pd.DataFrame | pd.Series, 
            'y_test': pd.DataFrame | pd.Series 
        }
     """
    # validate inputs 
    validate_input(features, pd.DataFrame, 'features') 
    validate_input(target, (pd.DataFrame, pd.Series), 'target') 
    validate_input(test_ratio, float, 'test_ratio') 

    # ensure features and target have identical indexes 
    if len(features.index) != len(target.index): 
        raise ValueError(f"Features and target are of unequal length: {len(features.index)}, {len(target.index)}")
    if not all(features.index == target.index): 
        raise ValueError("Features and target have misaligned indexes.") 

    # determine size of test_ratio relative to features and target index length 
    test_sample_size = round(len(features.index) * test_ratio) 
    # sample test data at random from features and target 
    test_idx = np.random.choice(features.index.to_numpy(), test_sample_size, replace=False) 
    # data_idx will be all the indices of features.index not in test_idx 
    data_idx = features.loc[~features.index.isin(test_idx)].index 
    # use the same strategy again to sample validation index from data_idx 
    val_idx = np.random.choice(data_idx, test_sample_size, replace=False) 
    # again, train_idx will be indices of data_idx not in val_idx 
    train_idx = np.array([idx for idx in data_idx if idx not in val_idx])
    # assert that val_idx and test_idx are of the same size 
    assert len(val_idx) == len(test_idx), f"Dev error: val_idx and test_idx were of unequal lengths: {len(val_idx)} != {len(test_idx)}" 
    # assert that val_idx and test_idx have no overlapping indices 
    assert all(val_idx != test_idx), f"Dev error: val_idx and test_idx have overlapping indices: {np.where(val_idx == test_idx)}"  
    # assert that train_idx and val_idx have no overlapping indices 
    assert all(train_idx != val_idx), f"Dev error: train_idx and val_idx have overlapping indices: {np.where(train_idx == val_idx)}" 
    # assert that train_idx and test_idx have no overlapping indices 
    assert all(train_idx != test_idx), f"Dev error: train_idx and test_idx have overlapping indices: {np.where(train_idx == test_idx)}"

    # Split sets according to randomized indices 
    X_train = features.loc[train_idx].copy() 
    y_train = target.loc[train_idx].copy() 
    X_val = features.loc[val_idx].copy() 
    y_val = target.loc[val_idx].copy() 
    X_test = features.loc[test_idx].copy() 
    y_test = target.loc[test_idx].copy() 

    return {
        'X_train': X_train, 
        'y_train': y_train, 
        'X_val': X_val, 
        'y_val': y_val, 
        'X_test': X_test, 
        'y_test': y_test 
    }


                

# ===  HELPER FUNCTIONS === 

# Helper for loading object based on extension type 
def load_set(path, ext): 
    if ext == 'csv': 
        return pd.load_csv(path) 
    elif ext == 'npy': 
        return np.load(path) 
    elif ext == 'npz': 
        return sparse.load_npz(path) 
    elif ext == 'json': 
        with open(path, 'r') as f: 
            return json.load(f) 
    else:
        return False

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
    
# validate correct inputs 
def validate_input(obj, valid_classes: str | tuple, param_name: str, error=TypeError, error_message=''): 
    """
    Validates that obj is of a type in valid_classes. 

    Parameters
    _________________________________________________________________________________________________
    obj: any Python object 
        Object to validate 
    valid_classes: str or tuple of Python object types 
        Classes or object types considered valid for obj 
    param_name: str 
        Name of parameter to validate 
    error: Error, default TypeError 
        Error(s) to raise if obj not valid 
    error_message: str 
        Custom error message to raise 

    Returns 
    _________________________________________________________________________________________________
    True if obj in valid_classes, otherwise errors are raised with error_message.
    """
    # construct default error message if none given 
    if not error_message: 
        error_message = f"Expected {valid_classes} for {param_name}, got {type(obj)}" 
    
    # validate input 
    if not isinstance(obj, valid_classes): 
        raise error(error_message) 
    else:
        return True 