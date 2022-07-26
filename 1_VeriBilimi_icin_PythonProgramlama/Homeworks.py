#####################################################################
# Fonksiyonlara Özellik ve Docstring Ekleme
#####################################################################

# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun. Var olan
#        özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

def grab_col_names(dataframe, cat_th=10, car_th=20): # cat_th: categorical threshold. Bir eşik değeri belirttik ve bundan az olan sayısal unique değerler bizim için birer kategorik değişkendir diyeceğiz
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişlenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.

    cat_th: int, float
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri

    car_th: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değiken listesi

    num_cols: list
        Numerik değişken listesi

    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi


    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde

    """

    # kategorikler
    cat_cols = [col for col in df.columns if df[col].dtypes in ["object", "category", "bool"]] # kategorik değişkenler alındı
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]] # numerik gibi gözüken ama aslında kategorik olanlar alındı
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]] # Toplam unique değeri 20'den fazla olan ve tipi kategorik veya obje olan (kısaca kardinalitesi yüksek olan) değişkenler alındı
    cat_cols = cat_cols + num_but_cat # cat_cols içinde tüm kategorik değer taşıyan veriler toplandı
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # cat_cols içinden kardinalitesi yüksek olan değişkenler çıkartıldı

    # sayısallar
    num_cols = [col for col in df.columns if str(df[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}") # Gözlem sayısı
    print(f"Variables: {dataframe.shape[1]}")  # Değişken sayısı
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False, null_val=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        }))
    print("######################################################")

    if null_val:
        print("Null Values sum..: ",dataframe[col_name].isnull().sum())
    if plot:
        sns.countplot(x=dataframe[col_name], data =dataframe)
        plt.show(block=True)


cat_summary(df, cat_cols[3], null_val=True)


# Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi (uygunsa) barındıran numpy tarzı docstring
# yazınız. (task, params, return, example)


def check_df(dataframe, head=5):
    """
    This function checks the dataframe and give us about
    dataframe's info.

    Parameters
    ----------
    dataframe: dataframe
        Dataframe to giving for function

    head: int (default = 5)
        Number of how many row that we want to see

    Returns
    -------
    This function returns nothing. It prints information

    Examples
    -------
     check_df(dataframe) or check_df(dataframe, head=10)

    """
    print("################### Shape ###################")
    print(dataframe.shape)

    print("\n################### Types ###################")
    print(dataframe.dtypes)

    print("\n################### Head ###################")
    print(dataframe.head(head))

    print("\n################### Tail ###################")
    print(dataframe.tail(head))

    print("\n################### NA ###################")
    print(dataframe.isnull().sum())

    print("\n################### NAs ###################")
    print(dataframe.describe().T)


def cat_summary(dataframe, col_name, plot=False, null_val=False):
    """
    Gives a summary of categorical values

    Parameters
    ----------
    dataframe: dataframe
        dataframe that we want to see summary
    col_name: list
        List of column names of categorical values
    plot : bool
        plot that we want to see a countplot about value. Default=False
    null_val: bool
        if true, then it gives sum of null values in the column. Default=False

    Returns
    -------
    It prints nothing

    Examples
    -------
    cat_summary(dataframe, col_name, plot=True, null_val=True)
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        }))
    print("######################################################")

    if null_val:
        print("Null Values sum..: ",dataframe[col_name].isnull().sum())
    if plot:
        sns.countplot(x=dataframe[col_name], data =dataframe)
        plt.show(block=True)


#####################################################################
# Pandas Alıştırmalar
#####################################################################

# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz
col_nunique = []
cols = []
for col in df.columns:
    col_nunique.append(df[col].nunique())
    cols.append(col)

pd.DataFrame(list(zip(cols, col_nunique)),
               columns =['column', 'nunique'])

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].value_counts()

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz
df[["pclass", "parch"]].nunique()

# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df.info()
df["embarked"] = df["embarked"].astype("category")
df.info()

# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == 'C']

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != 'S']

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"] < 30) & (df["sex"] == 'female')]
