
import numpy as np
import pandas as pd
import researchpy as rp

#
pd.set_option('display.max_columns', None)
#
pd.set_option('display.max_rows', None)
#
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#
pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x:'%.2f' % x)
pd.set_option('display.width',1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_seq_items', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.colheader_justify', 'right')
pd.set_option('display.html.table_schema', True)
pd.set_option('display.unicode.east_asian_width', True)

# EXCEL FILES
def read_all_excel_files(filename:str) -> list:

    from os import listdir

    files = listdir(filename)

    file_list = []

    for i in files:
        if "xlsx" and "xls" in i:
            file_list.append(i)

    return file_list

def read_all_excel_sheets(filename: str) -> print:

    for i in pd.ExcelFile(filename).sheet_names:
        print("Sheet Name: " + i)
        print("#" * 100)
        print(pd.read_excel(filename, sheet_name = i).head())
        print("#" * 100)

    # file_list = read_all_excel_files(r"C:\Users\bunya\Desktop\sun4tech\HEIMDALL")

    # for i in file_list:
    # print("\n" * 2)
    # print("#" * 100)
    # print("File Name: " + i)
    # print("#" * 100)
    # read_all_sheets(i)


# https: // stackoverflow.com / questions / 17977540 / pandas - looking - up - the - list - of - sheets - in -an - excel - file

# xl = pd.ExcelFile('foo.xls')

# xl.sheet_names  # see all sheet names

# xl.parse("sheet_name")  # read a specific sheet to DataFrame

# CSV FILES
def read_all_csv_files(filename:str) -> list:

    from os import listdir

    files = listdir(filename)

    file_list = []

    for i in files:
        if "csv" in i:
            file_list.append(i)

    return file_list

# Data info

def data_info(data, head = 5, tail = 5):

    print("\n", "#" * 25, "Head" , "#" *25)
    print(data.head(head), "\n")

    print("#" * 25, "Tail" , "#" *25)
    print(data.tail(tail), "\n")

    print("#" * 25, "Missing Values", "#" *25)
    print(pd.DataFrame(data.isnull().sum()), "\n")

    print("#" * 25, "Data Types", "#" *25)
    print(pd.DataFrame(data.dtypes), "\n")

    print("#" * 25, "Shape", "#" *25)
    print(data.shape, "\n")

    print("#" * 25, "Columns", "#" *25)
    print(list(data.columns), "\n")

    print("#" * 25, "Categorical Variables Summary", "#" *25)
    print(rp.summary_cat(data.select_dtypes(exclude=np.number)), "\n")

    print("#" * 25, "Continuous Variables Summary", "#" *25)
    print(rp.summary_cont(data.select_dtypes(include=np.number)), "\n")

    print("#" * 25, "Quantiles", "#" *25)
    print(data.quantile([0, 0.05, 0.25, 0.95, 0.99, 1]).T, "\n")

if __name__ == "main":

    # sys.path.insert(0, r"C:\Users\bunya\Desktop\BE_PC\3 -DataScience\1 - Codes\1 - Modül")

    # import be

########################################################################################################################
# Türkçe Karakter
# UnicodeEncodeError: 'charmap' codec can't encode characters
########################################################################################################################

"""

UnicodeEncodeError: 'charmap' codec can't encode characters

https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters

encoding="ISO-8859-1"
encoding="latin-1"
encoding="utf-8"
encoding="utf8"

"""

"""
pip3 freeze > requirements.txt  # Python3
pip freeze > requirements.txt  # Python2

"""