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
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/diabetes.csv")
    return data
df = load()

###############################
# Görev 1: Keşifçi Veri Analizi
###############################

# Adım 1: Genel resmi inceleyiniz

df.describe().T
df.shape
df.info()
df.head()

df.isnull().sum()


# Adım  2: Numerik ve Kategorik değişkenleri yakala

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # for cat_cols and cat_but_car cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =="O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in  dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız
df[cat_cols].value_counts()

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

for col in num_cols:
    print(df.groupby("Outcome").agg({col: "mean"}))
    #print(df.groupby("Outcome")[col].mean())


# Adım 5: Aykırı gözlem analizi yapınız

sns.boxplot(data = df)
plt.xticks(rotation=45)
plt.show()

def outliers_threshold(dataframe, col_name, q1=0.2, q3=0.9):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)

    iqr = quantile3 - quantile1

    up = quantile3 + 1.5 * iqr
    low = quantile1 - 1.5 * iqr

    return up, low

up, low = outliers_threshold(df,"Pregnancies")


def check_outlier(dataframe, col_name):
    up, low = outliers_threshold(dataframe, col_name)

    if ((dataframe[col_name] > up) | (dataframe[col_name] < low)).any():
        return True

    else:
        return False

for col in num_cols:
    print(col + "..:"+ str(check_outlier(df, col)))


# Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_names=False):
    na_list = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = (dataframe.isnull().sum()).sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    miss_df = pd.concat([n_miss, ratio], axis = 1, keys = ["n_miss", "ratio"])
    print(miss_df)

    if na_names:
        return na_list

missing_values_table(df)

na_cols = missing_values_table(df, True)

def missing_values_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(dataframe[col].isnull(), 1, 0)

    na_flags = [col for col in temp_df.columns if "_NA_" in col]

    for col in na_flags:

        print(col+"\n")
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                      "COUNT": temp_df[col].value_counts()}), end="\n\n\n")


missing_values_target(df, "Outcome", na_cols)


# Adım 7: Korelasyon analizi yapınız.
df.head()
corr = df[num_cols].corr()



###############################
# Görev 2 : Feature Engineering
###############################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df = load()
def zero_to_NaN(dataframe, columns):

    for col in columns:
        # zero_index = dataframe[dataframe[col] == 0].index
        # dfNan_but_zero[col] = np.where(dfNan_but_zero[col].isin(zero_index), "NaN",dfNan_but_zero[col])
        dataframe[col] = np.where((dataframe[col] == 0) | (dataframe[col] == "0"), np.NaN,
                                       dataframe[col])

    return dataframe

zeroNan = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

df = zero_to_NaN(df, zeroNan)
df.isnull().sum()

na_cols = missing_values_table(df, True)
missing_values_target(df, "Outcome", na_cols)

# NaN Values
cat_cols, num_cols, cat_but_car = grab_col_names(df)

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

df.isnull().sum()
df.head()


# Outlier Values : LOF

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(xlim = [0, 20], style = ".-") # 20 tanesini gözlemdeki, istersek 50 gözlem de yapabiliriz bunu
plt.show()

th = np.sort(df_scores)[2]
df[df_scores < th].index

df.shape
df = df.drop(df[df_scores < th].index, axis=0)
df.shape


# Adım 2: Yeni değişkenler oluşturunuz

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
corr = df[num_cols].corr()

df["Ins/Glu"] = df["Insulin"] - df["Glucose"]

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

# Encoding işlemine gerek yoktur

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()

# Adım 5: Model oluşturunuz.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)