###########################################################################
# FEATURE ENGINEERING & DATA PRE-PREOCESSING
###########################################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# !pip install missingno
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

# Her seferinde tekrar tekrar veri okuma işlemi yapılmasın diye aşağıdaki fonksiyonlar yazıldı

def load_application_train():
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/application_train.csv")
    return data

dff = load_application_train()

def load():
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/titanic.csv")
    return data

df = load()

#####################################################
# 1. Outliers (Aykırı Değerler)
#####################################################

####################################
# Aykırı Değerleri Yakalama
####################################

#########################
# Grafik Teknikle Aykırı Değerler
#########################
sns.boxplot(x=df["Age"])
plt.show()

#########################
# Aykırı Değerleri Nasıl Yakalanır
#########################
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr # Yaş değişkeninde 0 ın altında değer olmadığı için low değeri 0'ın altında çıkar

df[(df["Age"] < low) | (df["Age"] > up)]
df[(df["Age"] < low) | (df["Age"] > up)].index #Outlier'ların indexleri

#########################
# Aykırı Değer Var mı Yok mu Kontrolü
#########################
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

df[(df["Age"] < low)].any(axis=None) # low değerden aşağıda değer var mı diye sorguladık ve False geldi. Bu bool yaklaşımlar ilerde işimize yarayacak


#########################
# Fonksiyonlaştırma
#########################
# Outlier'lar için üst limit ve alt limiti belirleyen fonksiyon
def outliers_threshold(dataframe, col_name, q1=0.25, q3=0.75): # Temel literatürde 0.25-0.75 ama bunlar değiştirilebilir. Vahit hoca kullanırken 0.5-0.95 veya 0.1-0.99 olarak kullanıyormuş
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

low, up = outliers_threshold(df, "Age")

# Kolonda outlier var mı yok mu kontrolü yapan fonksiyon
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None): # Outliers varsa True dönmek amaçlı
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")


# Dataframe'deki tüm sayısal değişkenleri, kategorik değişkenleri, kategorik görünümlü kategorik olmayan değişkenleri, sayısal görünümlü kategorik değişkenleri getirecek olan fonksiyon

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Gives the values of names that categorical, numerical, categorical but cardinal
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ----------
        dataframe: dataframe
            The dataframe from which variable names are to be retrieved
        cat_th: int, optional
            Threshold value for numerical but categorical variables
        car_th: int, optional
            Threshold value for categorical but cardinal variables

    Returns
    -------
        cat_cols: list
            Categorical variable list

        num_cols: list
            Numerical variable list

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")  # Gözlem sayısı
    print(f"Variables: {dataframe.shape[1]}")  # Değişken sayısı
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# SibSp: 1. dereceden akrabalar, Parch: Nispeten daha uzak olan akrabalar
# Bu 2 kolon da threshold sonucu cat_cols listesine girdi. Bu değişkenleri kategorik veya sayısal olarak seçmek bize kalmış

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"] # ID ile ilgili olan kolonu çıkartmiş olduk

for col in num_cols:
    print(col, check_outlier(dff, col))


#########################
# Aykırı Değerlerin Kendine Erişmek
#########################

# 10 dan fazla outliers varsa head() kısmını görelim, 10 dan az ise tüm outliersları getirsin

def grab_outliers(dataframe, col_name, index=False):
    low,  up = outliers_threshold(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())

    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])


    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)


####################################
# Aykırı Değer Problemini Çözmek
####################################

#########################
# Silme
#########################

low, up = outliers_threshold(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape #Aykırı olmayanları seçdik

# Bu işlemi foksiyonlaştıralım
def remove_outliers(dataframe , col_name):
    low_limit, up_limit = outliers_threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe["Fare"] < low_limit) | (dataframe["Fare"] > low_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols  = [col for col in num_cols if col not in "PassengerId"]

df.shape
for col in num_cols:
    new_df = remove_outliers(df, col)

df.shape[0] - new_df.shape[0]


#########################
# BaskılamaYöntmii (re-assignment with threshold)
#########################

# Sildiğimmiz outlier'lar sebebiyle diğer işe yarar girdileri de silmek durumunda kaldık
# Genelde silmek yerine baskılama yöntemini kullanırız

low, up = outliers_threshold(df, "Fare")
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
# veya
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]


df.loc[(df["Fare"] > up), "Fare"] = up # Sol kısım bize eşik değerden yüksek olan "Fare" değerlerini verir. Bunu da üst limite eşitledik
df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_threshold(dataframe, variable)
    dataframe.loc[(df[variable] > up_limit), variable] = up_limit
    dataframe.loc[(df[variable] < low_limit), variable] = low_limit

# Veri setini tekrar yükleyip deneyelim
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

# Tekrar sorgularsak False gelecektir

for col in num_cols:
    print(col, check_outlier(df, col))


####################################
# Çok Değişkenli Aykırı Değer Analizi (Local Outlier Factor)
####################################

# 17 yaş veya 3 kez tekrar evlenme durumları aykırı değer değildir. Lakin 17 yaşındaki birisinin 3 kez evlenmiş olma durumu aykırı bir değerdir
# Buna çok değişkenli etki denir

# Lof yöntemi gözlemleri bulundukları konumda, yoğunluk tabanlı skorlayarak buna göre aykırı değer tanımı yapabilmemizi sağlar
# Veri çok değişkenli ise pca yöntemi ile olabildiğince değişkenleri indirgeyip 2 boyutlu bir grafikte outlier görselleştirmesi yapabiliriz

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include = ["float64", "int64"])
df = df.dropna()
df.head()

# df'teki outlier'ları kontrol etme
for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outliers_threshold(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape #sadece "carat" değişkeninde 1889 outlier var. Diğer değişkenlerde de ha keza buna benzer değerler var.

# Biz eğer bunların hepsini silseydik büyük bir veri kaybı yaşardık. Yahut bunları baskılasaydık da veri içinde gürültüye sebep olacaktık

# Lof Yönünden İnceleyelim:
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

# Buradaki değerler genel anlamda -1 çıktı. O halde diyeceğiz ki -1'den uzaklaştıkça (mesela -10) bu değer outlier'dır

np.sort(df_scores)[0:5] # En kötü 5 gözlem. Yani bu değerler df'teki gözlemlerin konumlandırırlıp verdiğimiz neighbors değerine göre ne kadar aykırı olduğunu bize söylüyor
# df_score değerleri 1 lerde olsaydı bu sefer de mesela 10'a yaklaştıkça outlier olma eğilimnde derdik
# Burada gelen sonuçlardaki -8 li değerler de outlier olma eğiliminde olduğunu göstermekte

# Şimdi bir eşik değer belirleyerek lof'u bu değerden fazla olanları atalım
# Bu eşik değerini belirlemek için pca'de kullanılan elbow(dirsek) yöntemini kullanalım

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stcked = True, xlim = [0, 20], style = ".-") # 20 tanesini gözlemdeki, istersek 50 gözlem de yapabiliriz bunu
plt.show() # Bu plotta çıkan her bir nokta bir threshold adayıdır
# Plot'u incelersek EĞİMİN en bariz şekilde değiştiği yer seçeceğimiz threshold dur. Burada da 3. indexteki yer bizim için uygundur

th = np.sort(df_scores)[3]

df[df_scores < th] # Bize outlier olan gözlemleri döner. Bunlar 3 tanedir. Böylece binlerce veri silmekten kurtulduk

# Peki bu değerler neden aykırıdır. Bakalım:
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# Burada outlier olanların indexini alıp silebiliriz
df[df_scores < th].drop(axis=0, labels = df[df_scores < th].index)

# Baskılama yapabilirdik ama ne ile ? Lof yöntemi sonucu hangi faktörün bu gözlemleri outlier yaptığını bilmiyorum
# Outlier olmayan bir gözlem ile değiştirebiliriz. 3 tane gözlem için kopyalamak sorun olmaz. Ama ya 100 tane olsaydı. 100 tane aynı veriden koymuş olurduk. Veriyi zorla bozmuş olurduk

# Ağaç yöntemleri kullanıyorsak outlier'lara hiiç dokunmayacağız. Ama illa ki silelim istiyorsak çok küçük thresholdlar ile bu işlemleri yapmalıyız
# iqr hesaplayıp 025-075 değil de 0.1-0.9 gibi değerlerle bu işlemleri yapmalıyız


#####################################################
# 2. Missing Values (Eksik Değerler)
#####################################################

####################################
# Eksik Değerlerin Yakalanması
####################################

# Eksik gözlem var mı yok mu sorgusu
df.isnull().values.any()

# Değişkenlerin eksik değer sayısı
df.isnull().sum()

# Değişkenlerin tam değer sayısı
df.notnull().sum()

# Veri setindeki toplam eksik değer sayısı. Gözlemde en az 1 tane NA varsa onu da saydığı için 866 tane döndü
df.isnull().sum().sum()

# Eksiklik içeren gözlem birimleri
df[df.isnull().any(axis=1)]

# Tan olan gözlem birimleri
df[df.notnull().any(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending = False)

# Her bir değişkendeki eksiklik oranları
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# Sadece eksik değerleri seçme
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# Bu işlemleri fonksiyonlaştırırsak
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe.isnull().sum().sort_values(ascending = False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)

####################################
# Eksik Değer Problemini Çözmek
####################################

missing_values_table(df)

#########################
# Çözüm 1: Hızlıca Silme
#########################
df.dropna().shape


#########################
# Çözüm 2: Basit Atama Yöntemleri
#########################

df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())

# Bunları tek tek değil de tüm kolonları fonksiyonlarştırarak doldurmak istersek;
df.apply(lambda x: x.fillna(x.mean()), axis=0) # Sadece bu hata verir. Object'lerin ortalaması olmaz
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

# Kategorik değişkenler nasıl doldurulur ?
df["Embarked"].fillna(df["Embarked"].mode()[0])  # Mode ile doldurmak en çok tercih edilenlerdendir

# Bunu da fonksiyonlaştıralim
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0) # Kardinal değişkenleri doldurmamak için 10 dan küçük ve eşitler için doldurduk

#########################
# Kategorik Değişken Kırılımında Değer Atama
#########################
df.groupby("Sex")["Age"].mean() # male ortalaması 30.727 çıkar
df["Age"].mean() # Ortalama 29.699 çıkar

# Velhasıl biz tüm kolonun ortalaması şeklinde değil de yukarıda daha mantıklı olarak gözüken
# kırılımlar ile doldurma yaparsak daha sağlıklı olacaktır
# Peki bunu programatik olarak nasıl yaparız ?

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")) # transform sayesinde ortalama ile verileri yerleştirmiş olduk
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum() # 0 gelir

# loc ile deneyelim
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

#########################
# Çözüm 3: Tahmine Dayalı Atama
#########################

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# cat_cols ları sayısal ifade edebilmek için one-hot encoder kullanalım
df2 = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) # drop_first 2 sınıfa da encoder yapmaz da birini düşürmemizi sağlar
# get_dummies sadece kategorik değişkenlerde işe yaradığı için num_cols'u da rahatlıkla ekledik

df2.head()

# Değişkenlerin standartlaştırılması: Tahminleri ML algortimları ile yapacağımızdan dolayı standartlaştırıyoruz
scaler = MinMaxScaler() # 0-1 arasına haritalar
df2 = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns)
df2.head()

# Knn'in uygulanması: ML yöntemi ile tahmine dayalı şekilde boşlukları doldurmamızı sağlayacak
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) # Bir gözlemin yaş kısmı boş olsun. knn gidip diğer gözlemleri ile beraber en çok benzeyen 5 gözlemi alacak. Onların yaş değişkenlerinden yola çıkarak yaşı dolduracak
df2 = pd.DataFrame(imputer.fit_transform(df2), columns=dff.columns)
df2.head()

# Standartlaştırdığımız değerleri normal haline çevirelim
df2 = pd.DataFrame(scaler.inverse_transform(df2), columns=df2.columns)

# Peki nereye ne atamış bu tahmine dayalı model onu nasıl ölçeriz?
# Orjinal dataframe'e değer atanmış olan 2. dataframe'deki kolonu ekleriz. Sonra da karşılaştırırız

df["age_imputed_knn"] = df2["Age"]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

df.Cabin.nunique()

##############################
# Gelişmiş Analizler
##############################

#########################
# Eksik Verinin Yapısının İncelenmesi
#########################
msno.bar(df) # İlgili veri setindeki değişkenlerdeki tam olan gözlemleri gösterir
plt.show()

msno.matrix(df) # Değişkenlerdeki eksikliklerin birarada meydana geliyorsa oluşan siyah beyaz çizgiler diğer kolon ile benzerlik gösterir
plt.show()

msno.heatmap(df) # Eksik değerler bir korelasyonlarla (yani eksiklikler birlikte mi oluşuyor) gerçekleşiyorsa bunları verir
plt.show()

# Bizim veri setimiz için değerlendirirsek oluşan eksiklikler birlikte ortaya çıkma eğiliminde değildir

#########################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
#########################
df = load()
missing_values_table(df, True)
na_cols = missing_values_table(df, True) # Eksik değerleri aldık

# Bir fonksiyon yazacağız ve şu şekilde çıktı verecek
"""
Fonksiyon, içerisinde na olan kolonları alacak. Burada na olan değerlere 1, 
na olmayan değerlere 0 vererek bir temsil etme işleminde bulunacak. Daha sonra
bu na olma ve olmama durumunun bizim tahmin etmek istediğimiz durum olan 
'survived' değişkenine olan etkisine bakacak. Örnek verelim;
Cabin değişkeninde na olan değerlere sahip bireyler %30 oranında hayatta kalmış iken
na olmayanlar %60 oranında hayatta kalmış. Bu çok büyük bir değerdir bizim için
Bu durumda Cabin değişkeni silinip atılabilecek bir kolon gibi duruyorken artık işler değişti

Şöyle bir durum da var ki ya na olan 2 taneyse ve bu sebeple na olanların daha çok 
hayatta kalma yüzdesi varsa? O yüzden de bunlara ait count değerlerini de fonksiyonda bastırdık 
"""

def missing_values_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0) # null gördüğün yerlere 1 geri kalanlarına 0 yaz demektir

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns # dataframe'e eklediğimiz kolonlardan içerisinde _NA_ olanları yani na içerikli olanları alacak. Tüm satırları da alacak önce

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n") # Bunun yerine temp_df["col"].value_counts() yazılabilir miydi acaba. Bir ara dene

missing_values_target(df, "Survived", na_cols)