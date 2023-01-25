# unstructured class
import time

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


########################################################################################################################
########################################################################################################################
# Rich text format to text
########################################################################################################################
########################################################################################################################

def rft2text(data, column, inplace: bool = False, isnull: bool = False, suffix: str = None, prefix: str = None, error: str = "strict"):

    from striprtf.striprtf import rtf_to_text
    # pip install striprtf
    # https://pypi.org/project/striprtf/

    """
    To convert values in a column of a Dataframe that are in rich text format to plain text for each row, 
    you can use this function and striprtf library.

    Note: Creates a new column.
    Note: it uses the striprtf library.
    
    :param data: dataframe as pandas dataframe
    :param column: column name to apply as string
    :param inplace: same as
    :param isnull: if column has null values, should change with True
    :param suffix: add suffix to new column name as string
    :param prefix: add prefix to new column name as string
    :param error: if you dont want to errors during to convers such as unicode, enter ignore as string
                  but it may cause data loss
                  How to handle encoding errors. Default is "strict", which throws an error. 
                  Another option is "ignore" which, as the name says, ignores encoding errors.

    :return: dataframe

    library:
    from striprtf.striprtf import rtf_to_text
    # pip install striprtf
    # https://pypi.org/project/striprtf/
    """

    '''
    If you are receiving the error message "UnicodeEncodeError: 'charmap' codec can't encode character '\xdd' in position 0: character maps to <undefined>", it means that there is a character in the decode table that is not defined. In this example, it is "\xdd". This error indicates that the character in question is not supported by the 'charmap' codec.

    To resolve this issue,
    you can try adding the character in question to the algorithm/function that you are using, or you can edit the source file (cp1253.py) and add the character to it. My suggestion would be to add the character to the algorithm.
    It would be better not to play around with the source file.
    Or you can activate errors params from inside of rtf_to_text function but,
    it may cause data loss

    Also you may need to change encoding:

    UnicodeEncodeError: 'charmap' codec can't encode characters

    https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters

    encoding="ISO-8859-1"
    encoding="latin-1"
    encoding="utf-8"
    encoding="utf8"
    '''

    if inplace:
        data = data.copy()

    if isnull:
        data[column].fillna("", inplace=True)

    # Create the new column name
    if suffix:
        new_col = data[column].name + suffix
    elif prefix:
        new_col = prefix + data[column].name
    else:
        new_col = data[column].name + "_rft2text"

    data[new_col] = data[column].apply(lambda x: rtf_to_text(x, errors=error))

    return data

########################################################################################################################
########################################################################################################################
# Asynchronous programming
########################################################################################################################
########################################################################################################################

if __name__ == "main":

    print(__name__)

    import threading as th
    import time

    data1 = {"A": [1, 2, 1],
                 "B": ["Z", "X", "Z"],
                 "C": ["A", "S", "D"]}

    data1 = pd.DataFrame(data1)

    data2 = {"ID": [1, 1, 2, 2, 3, 4, 5],
            "TEST": ["1.45", "2.45", "47", "12.12.00", "123.00.00", "23.45", "215.00"]}

    data2 = pd.DataFrame(data2)

    def sum(a,b):
        print("Start")
        time.sleep(5)
        print(a+b)

    def end ():
        print("End")


    th.Thread(target=cat_to_var, args=(data1, "B", "C")).start()
    th.Thread(target=del_repeated_last_occur_str, args=(data2, ".", "TEST")).start()

    th.Thread(target=sum, args=(10,10)).start()
    th.Thread(target=end, args=()).start()

########################################################################################################################
########################################################################################################################
# Parallel Programming
########################################################################################################################
########################################################################################################################

if __name__ == "main":

    from joblib import Parallel, delayed


########################################################################################################################
########################################################################################################################
# multiprocessing
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Subprocess
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
# Decorators
########################################################################################################################
########################################################################################################################





########################################################################################################################
########################################################################################################################
# RFT (Rich Text Format) to Text
########################################################################################################################
########################################################################################################################

from striprtf.striprtf import rtf_to_text
# pip install striprtf
# https://pypi.org/project/striprtf/





