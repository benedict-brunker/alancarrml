if test_ratio is not None and test_ratio > 0: 
    # two or three way split 
    if val_ratio is not None and val_ratio > 0: 
        #  three-way split 
        if by_time and time_col: 
            # split by time 
            if pop: 
                # pop target and return 
            else: 
                # return without popping target 
        else: 
            # split by simple indexing  
            if pop: 
                # pop target and return 
            else: 
                # return without popping target 
    elif by_time and time-col: 
        #  two-way train-test split by time 
        if pop: 
            # pop target and return 
        else: 
            # return without popping target 
    else: 
        # return two-way train-test split with simple indexing 
        if pop: 
            # pop target and return 
        else: 
            # return without popping target 

elif val_ratio is not None and val_ratio > 0: 
    # return two-way train-val split 
    if by_time and time_col: 
        # split by time 
elif pop: 
    # pop target and return 
else: 
    # just return with warning message  


    # for testing 

assert all(X_data.index != X_test.index), "Implementation error: X_data and X_test shared indexes." 
assert all(y_data.index != y_test.index), "Implementation error: y_data and y_test shared indexes." 

