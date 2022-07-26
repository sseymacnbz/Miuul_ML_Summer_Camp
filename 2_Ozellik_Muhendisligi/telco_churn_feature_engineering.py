import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/Telco-Customer-Churn.csv")
    return data

df = load()

###############################
# Görev 1: Keşifçi Veri Analizi
###############################

# Adım 1: Genel resmi inceleyiniz

def df_summary(dataframe):
    print("************ HEAD ************")
    print(dataframe.head(), end="\n\n")
    print("************ SHAPE ************")
    print(dataframe.shape,end="\n\n")
    print("************ DESCRIBE ************")
    print(dataframe.describe().T,end="\n\n")
    print("************ INFO ************")
    print(dataframe.info(), end="\n\n")
    print("************ NULLs ************")
    print(dataframe.isnull().sum(),end="\n\n")

df_summary(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
sns.boxplot(data=df[num_cols])
plt.show()

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

# num_cols:
for col in num_cols:
    print(df.groupby("Churn").agg({col: "mean"}),end="\n\n")

# cat_cols:
for col in cat_cols:
    print(col)
    print(df[col].value_counts(),end="\n\n")

"""
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    
    print("########################################################")
    
    if plot:
        sns.countplot(x=dataframe[col_name], data =dataframe)
        plt.show(block=True)

cat_summary(df, cat_cols) 
"""

# Adım 5: Aykırı gözlem analizi yapınız.



