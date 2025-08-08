# dates.py 

# Dependencies 

import pandas as pd 

# convert_to_date 
def convert_to_date(df: pd.DataFrame, cols: str | list, format=None, errors='raise'):  
    """
    Converts all columns of df in cols to datetime, returning transformed dataframe. 

    Parameters
    _____________________________________________________________________________________________________________
    df: pd.DataFrame 
        DataFrame to transform 
    cols: str | list
        [List of] column names to convert to datetime. 

    Returns 
    _____________________________________________________________________________________________________________
    pd.DataFrame: DataFrame with cols transformed to datetime. 
    """ 
    # validate inputs 
    from alancarrml.data.sets import validate_input

    validate_input(df, pd.DataFrame) 
    validate_input(cols, (str, list))  

    datetime_cols = {} # for storing datetime-transformed columns as {'col_name': datetime_column} pairs
    col = '' # Placeholders to allow formatted error_message string
    e = None 

    error_message = f"""
                  Column {col} could not be converted to datetime. 
                  This column will be skipped and returned untransformed. 
                  Error message: 
                  {e} 
                  """

    for col in cols: 
        try: 
            col_datetime = pd.to_datetime(df[col], errors=errors, format=format) 
            datetime_cols[col] = col_datetime
        except pd.errors.ParserError as e: 
            print(error_message) 
            continue
        except ValueError as e: 
            print(error_message) 
            continue 
    
    # if datetime_cols remains empty, alert the user that transformations were unsuccessful 
    if not datetime_cols: 
        print(f"None of {cols} could be succesfully transformed to datetime. Returning original df.") 
        return df 
    
    # replace original df with transformed columns 
    datetime_df = df.copy() 

    for col in datetime_cols.keys():
        datetime_df.loc[:, col] = datetime_cols.get(col) 
    
    return datetime_df 