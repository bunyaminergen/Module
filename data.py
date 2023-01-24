# unstructured class

import numpy as np
import pandas as pd

"""
This module contains functions that can be used when dealing with data cleaning for unstructured data 
"""

def cat_to_var(data, cat_column, var_column, suffix: str = None, prefix: str = None, inplace: bool = False ):

    """

    This function extracts the values of a categorical variable from another variable
    and creates a new column with those values, corresponding to the index of each category in the original variable.
    Let's say we have a variable called TEST that contains categorical variables representing test names.
    These test values are also present in another variable. We want to create a new column,
    where we assign the values from the other variable, corresponding to the index of each test name in the TEST variable.

    You can use this function at that time.

    You can test it before using it with the example below.
    # https://stackoverflow.com/questions/48027171/create-a-variable-in-a-pandas-dataframe-based-on-information-in-the-dataframe

    numpy where
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Test:

    dataframe = {"A": [1,2,1],
                 "B": ["Z", "X","Z"],
                 "C": ["A","S","D"]}

    data = pd.DataFrame(dataframe)

    cat_to_var(data,"B","C", suffix="_deneme", inplace=True)

    data.head()

    """

    # Create a copy of the dataframe if inplace is not set
    if not inplace:
        data = data.copy()

    # Iterate through unique values in the categorical column
    for j in data[cat_column].unique():
        # Create the new column name
        if suffix:
            new_col = j + suffix
        elif prefix:
            new_col = prefix + j
        else:
            new_col = j

        # Assign the values from the variable column to the new column
        data[new_col] = np.where(data[cat_column] == j, data[var_column], np.NaN)

    return data

def del_repeated_last_occur_str(data, char, column, inplace: bool = False):

    """

    The function is used to remove the last occurrence of a repeated string in a specific column of a DataFrame.
    If there are two repeated string expressions in the values of a column,
    the function will remove everything after the last occurrence of the specified character.

    :param data: dataframe as dataframe
    :param char: want to convert character as string
    :param column: want to implement apply columns as string
    :param inplace: same as other libraries
    :return: return data as dataframe

    Test and Example:

    data = {  "ID" :  [1,1,2,2,3,4,5],
            "TEST" :  ["1.45","2.45","47","12.12.00","123.00.00","23.45","215.00"]}

    data = pd.DataFrame(data)

    data.head()

    data["TEST"] = data["TEST"].astype(float)

    del_repeated_last_occur_str(data,".","TEST", inplace = True)

    The above code is a function to remove the last occurrence of a repeated string in a specific column of a DataFrame.
    When you want to convert the 'TEST' column to float, it will give an error because of unwanted characters.
    If you do not want to replace all the characters with methods like replace(),
    you can use this function to remove only the characters after the specific one in the column.
    Note: If you use pd.to_numeric(data["column_name"], errors='coerce') to convert the column to float,
    it will return 'nan' for invalid values and may cause data loss.
    This function can be used as a pre-processing step to remove unwanted characters before converting the column to float.


    # https://www.w3schools.com/python/ref_string_rfind.asp
    # https://www.w3schools.com/python/ref_string_rindex.asp
    # https://docs.python.org/3/library/stdtypes.html#str.rindex:~:text=str.rfind,is%20not%20found.
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html

    """
    # Create a copy of the dataframe if inplace is not set
    if not inplace:
        data = data.copy()

    if data[column].str.contains(char).any():
        for i in data.index:
            if data.loc[i, column].count(char) > 1:
                data.loc[i, column] = data.loc[i, column][:data.loc[i, column].rindex(char)]
    else:
        raise ValueError("The character is not present in the column values.")

    return data
