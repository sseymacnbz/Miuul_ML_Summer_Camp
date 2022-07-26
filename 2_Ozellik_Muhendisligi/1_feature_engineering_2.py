#####################################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#####################################################

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

def load_application_train():
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/application_train.csv")
    return data
dff = load_application_train()

def load():
    data = pd.read_csv("2_Ozellik_Muhendisligi/datasets/titanic.csv")
    return data
df = load()

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

####################################
# Label Encoding & Binary Encoding
####################################

df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

# Hangi sınıfa 0 hangi sınıfa 1 verildiğini unutursak şunu yapabiliriz
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Öncelikle encoding'i 2 sınıflı değişkenler için düşünelim (Yani ilk önceliğimiz binary durumdakiler için)
df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# Önemli Not: len(unique()) eğer değişkende NaN değerler varsa onları da sınıf olarak alır
# nunique() ise bunu yapmaz ve NaN'ları sınıf olarak değerlendirmez. NaN içerikli bir sınıfa len unique yaparsak 4 çıkyorsa, nunique yaptığımızda 3 çıkar

for col in binary_cols:
    label_encoder(df, col)

df["Sex"].head()

# Bunu daha büyük bir veri setinde denersek
dff = load_application_train()
dff.shape

dff[binary_cols].head()

binary_cols = [col for col in dff.columns if dff[col].dtype not in [int, float]
               and dff[col].nunique() == 2]
for col in binary_cols:
    label_encoder(dff, col)

# nunique() NaN'ları sınıf olarak almaz ama label_encoder'dan geçirilmiş binary değişkelere bakarsak
# 0-1'in yanında 2 değerini de görürüz. Label encoder NaN'lara da 2 atamıştır


####################################
# One-Hot Encoding
####################################

df = load()
df.head()
df["Embarked"].value_counts() # Nominal değerde bir  değişkendir. Değerleri arasında herhangi bir öncelik ilişkisi yoktur
# Bu sebeple embarked değişkenine label encoding değil one-hot encoding işlemi uygulayacağğız

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head() #Dönştürmek istediğimiz kolonlar ıda seçiyoruz. Böylece  ekstra bir seçme işlemimizi yapmamıza gerek kalmaz
# drop_first ilk sınıfı düşürür, bunun bir de dummy_na parametresi de vardir. Bu da eğer istersek na olan gözlemler için de bir kolon oluşturur

# Bunu Sex kolonu için de(yani binary olan kolonlar için de) yapabiliriz. Bir sıkıntı olmaz. drop_first kolonunu unutmamak lazım

# Bu işlemi fonksiyonlaştırırsak;
def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# grab_col_names fonksiyonunu çağırıp kategorik değişkenlere bir label encoding işlemi yapabiliriz
# Lakin şöyle  bir sıkıntı var ki grab_col_names ten gelecek olan kategorik kolonlarının içinde mesela
# survived de var ki bu bizim bağımlı değişkenimiz ve bunun encodinng olmmasına gerek yok.
# Veya başka binary olan değişkenler için yapmak istemiyor olabbilirz.
# Vahit Hoca bu durum için one hot encoding kolonlarıın ayrıyetten seçmemizin daha iyi olacağını ve kendisinin de bunu yaptığını söyledi

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()


####################################
# Rare Encoding
####################################
# Genelde model kurma süreçlerinde karmaşıklık ile değil basitlik ve genelleneblirlik ile ilgileniyoruz
# Yani önemli olan herkesi kapsayalım değil çoğunluğu kapsayalım

# Bir veri setinde unique değerlerinden birisinin frekansı 2 olsun. Biz one-hot encoding yaptığımızda bu 2 frekansa ait değişken için de
# bir kolon oluşturulacak ve belki de 100.000 gözlemli bir dataset için bir kolon oluşturucaz ve sadece 2 gözlem 1 olacak
# Velhasıl her ne kadar bu durum göreceli olsa da biz frekansı az olan değerleri bir araya toplayıp bir arada encoding yapabiliriz

# Bu encoding sürekli kullanılan bir  encoding değildir. Bir takım aşamalar sonucu
# encoding işlemlerimizi gerçekleştireceğiz;

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi _ Yani belki bu rare değişken bağımlıı değişkenimiz için anlam ifade ediyodur. Öyle olursa onu rare encoding e sokmayz
# 3. Rare encoder'ın yazılması


#######################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
#######################
df = load_application_train()
df["NAME_EDUCATIONN_TYPE"].value_counts() # academic degree değişkeni diğerlerine göre az

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("######################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data =dataframe)
        plt.show(block=True)
for col in cat_cols:
    cat_summary(df, col)   # Böylece tüm kategorik değişkenler üzerinde gezip değişkenlerin içerisindeki sınıfların frekans ve oralarını görebiliriz
# Mesela "Working" değişkenindeki 'Unemployed' sınıfından ititbaren hep binde 7 olacak şekilde veya binde 6 olacak şekilde değerler var

#######################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
#######################

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INNCOME_TYPE")["TARGET"].mean()

# Burada groupby'ladığımız kısımda çıkan değrlerde 0'a yakın olmak demek kredisini odemektedir anlamında
# 1e yakın olmak demek kredisini ödeyememek demektir
# value counts da baktığımız businessman ve unemployed sınıfları aslında diğerlerine nazaran düşüktür. Yan frekansları azdır. Bu 2si biraraya getirilebilir gibi düşünülmektedir
# Lakin TARGET değişkenine(bağımlı değişkenimiz) olan ortalamasına vurursak unemployed sınıfı 1 e yakın çıkarken
# businessman sınıfı 0 çıkmaktadır. Biz bu 2 sınıfı bir araya getirirsek gürültü eklemiş olabilriiz veya iyi bir şey yamı şda olabiliriz
# Bu kısımlar mayın tarlasıdır. Devamını getirmek bize kalmıştır ama biz şimdilik bu az frekansa sahip sınıfları nasıl birleştiririz ona bakalım

# Yukarda yaptığımız 2 işlemi bir araya getirecek olan bir fonksiyon yazalım
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN":dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#######################
# 3. Rare encoder'ın yazılması
#######################
def rare_encoder(dataframe, rare_perc): #rare_perc'in altında kalanlar rare sınıfına atanacak
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)] # Rare içerik barındıran kolonlar seçildi

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01) # %1'den az olanları rare yapar


####################################
# Feature Scalling (Özellik Ölçeklendirme)
####################################

#######################
# Standart Scaler
#######################

# Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s . Aykırı verilerden etkilenir
df = load()
ss = StandardScaler()
df["Age_standart_scaler"] = ss.fit_transform(df[["Age"]]) # Karşılaştıralım diye atama işlemi yaptık
df.head()


#######################
# Robust Scaler
#######################

# Medyanı çıkar, iqr'a böl. Böylece Standart Scaler'a göre aykırılıklardan daha az etkilenir

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T


#######################
# MinMax Scaler
#######################

# Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

# Oluşturulan 3 yeni kolonu karşılaştıralım

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

for col in age_cols:
    num_summary(df, col, plot=True)


#######################
# Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning
#######################

df["Age_qcut"] = pd.qcut(df["Age"], 5)


#####################################################
# 4. Feature Extraction (Özellik Çıkarımı)
#####################################################

# Var olan değişkenlerin içerisinden başka değişkenlerin çıkarılabilmesi durumudur

####################################
# Binary Features: Flag, Bool, True-False
####################################

df = load()
df.head()

# Cabin değişkeninde NaN olmayanlara 1, olanlara 0 yazalım. Bakalım bu kadar çok NaN
# içeren Cabin değişkeni gerçekten de çok mu anlamsız bir değişken

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int") # notnull() (Yani dolu mu diye soruyor) ile değişkenleri bool'a çevirdik. Sonra da bunları int'e çevirip 0 ve 1 lerden oluşan bir kolon elde ettik

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"}) # Cabin değişkeni dolu olanlar %66 oranında hayatta kalmış ama Cabin değişkeni NaN olanlar %30 oranında hayatta kalmış. Velhasıl bu değişken kesinlikle gereksiz ve silinmesi gereken bir alan değildir

# Bu Cabin değişkeni ile bağımlı değişkenimiz olan 'survived' değişkeni arasında daha derin bir bakış atalım. Oran testi yapalım:
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(), # Kabin numarası 1 olup yaşayan kaç kişi var
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()], # Kabin numarası 0 olup yaşayan kaç kişi var  # Bu 2 si bize başarı sayısını verecek

                                     nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],  # Kabin numarası olanlar kaç kişi
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]]) # Kabin numarası olmayanlar kaç kişi   # Bu da gözlem sayısı
# 2 parametre alır ilk verdiklerimiz başarı sayısıdır (count olan), 2. si ise frekanstır (nobs olan)

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Burada kabin numarası olmayanlar %30, olanlar %66 oranında hayatta kalmıştı ya, Bu iki değer p1 ve p2 olarak kabul edilip
# Yukarıdaki hipoteze gönderildi. Burada da pvalue değeri 0.05'ten küçük olduğundan dolayı da aralarında istatistiki olarak anlamlı bir farklılık vardır denilebilir
# Yine de emin değiliz. Çünkü buna sadece survived değişkeni etki etmiyor olabilir. Çok değişkenli olarak incelemedik


df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO" # SibSp yakın kişiler (çocuk, eş, anne vs), Parch ise biraz daha uzaktan akraba (teyze vs)  # Burada 2 değişkeni topladığımızda 0 dan buyukse yalnız mı kolonuna "NO" yazacak
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES" # Burada da akrabalarının toplamı sıfırsa bu kişi yalnız mı kolonunda yes olacak

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"}) #Burada çıkan sonuca göre yalnız olmayanlar %50 oranında hayatta kalmış
# Buna da bir proportion test yapalım

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(), # Kabin numarası 1 olup yaşayan kaç kişi var
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()], # Kabin numarası 0 olup yaşayan kaç kişi var  # Bu 2 si bize başarı sayısını verecek

                                     nobs = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],  # Kabin numarası olanlar kaç kişi
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))  # pvalue 0.05 in altında olduğundan hipotez reddedildi ve bu ikisi arasında bir bağlılık vardır dendi

####################################
# Text'ler Üzerinde Özellik Türetmek
####################################

df = load()

#######################
# Letter Count
#######################

# İsimlerdeki harfleri saydıracağız. Belki de kraliyet ailesinden gelen birisi olma durumu hayatta kalma olasılığını arttırmıştır.
# Bu şekilde farklı düşünceler bize diğer gerçek hayat problemlerinde fayda sağlayacaktır. Çok yönlü düşünmek lazım

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()


#######################
# Word Count
#######################

# Bir de kelimleri sayalım

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" "))) # İsmi komple bir string'e çevirip boşluklara göre ayırıp saysın

#######################
# Özel Yapıları Yakalamak
#######################

# Name kısmında bazı isimlerde "Dr" ibaresi var. Bunu içeren isimleri flagleyelim
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")])) # İlgili satırı split et. Sonra da her bir değer liste halinde erişilebilir olacaktır. Bu kelimelerde gez. Kelimelerde başlangıçta Dr ifadesi varsa onu seç ve len'ine bak.  # Bir isimde dr ifadesi 1 kere olacağından len'in çok da şeyi yok sanırım

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})  # Doktor olanların hayatta kalma oranı daha yüksek. Doktor count'ı ise 10 dur. Bu durum için direkt çok önemlidir diyemesek de yine de kenarda durmasında fayda vardır



####################################
# Regex ile Değişken Türetmek
####################################

# İsimlerin içerisindeki Mr. , Miss. , Mrs. gibi alanları almaya çalışacağız
# Regex'i kullanmak için bu ibarelerin nasıl bir patter'de olduğuna bakalım
# Büyük harf ile başlıyor. Sonunda nokta var ve ardından bir boşluk bırakılmış

df["NEW_TITLE"] = df["Name"].str.extract(" ([A-Za-z+)\.", expand=False) # extract: Çıkar. Önünde boşluk olacak, sonunda nokta olacak, büyük ya da küçük harflerden oluşacak ifadeleri al

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# Çok güzel bir çıktı geldi. Sadece miss, mrs'ten oluşmuyormuş. Çok daha farklı unvanlar geldi.
# Biz age için ortalama alıp atıyorduk ya. Şimdi ise ünvanlarına göre yaşlarının ortalamasını NaN'lara atamak daha mantıklı olacaktır


####################################
# Date Değişkenleri Türetmek
####################################

dff = pd.read_csv("2_Ozellik_Muhendisligi/datasets/course_reviews.csv")
dff.head()
dff.info()

# Timestamp değişkeninin tipini object'ten datetime'a çevirelim öncelikle
dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d") # Timestamp'in hangi sıralamada olduğunu da format parametresi ile vermeliyiz

# Yıl değişkeni türeteceksek
dff["year"] = dff["Timestamp"].dt.year

# Ay değişkeni türeteceksek
dff["month"] = dff["Timestamp"].dt.month

# Bugünün tarihinden veri setindeki yılları çıkartalım
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# 2 tarih arasındaki farkın ay cinsinden ifade edilişi
dff["month_diff"] = (date.today().year - dff["Timestamp"].df.year) * 12 + date.today().month - dff["Timestamp"].df.month # Önce yıl farkını heasplamalıyız ve kaç yıl farkı varsa onu 12 ile çarpmalıyız ve ay cinsine çevirmeliyiz. Daha sonra da normal ay farkına bakmalıyız

# Veri setindeki günlerin hangi günler olduğuna bakmak istersek
dff["day_name"] = dff["Timestamp"].dt.day_name()


####################################
# Feature Interactions (Özellik Etkileşimleri)
####################################
df = load()
df.head()

# Değişkenlerin birbiriyle etkileşime girme durumudur. Ör: Kolonu 3 ile çarp vs
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"] # yaşlar ile pclass ları çarparak kişilerin refah durumuna ilişkin özellik elde etmeye çalışıldı. Mesela genç yaşta yüksek sınıftaki birinin değeri yüksek çıkabilir

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 # Ailenin büyüklüğünü elde ettik. +1 ise seyahat eden kişinin kendisidir


df.loc[(df["Sex"] == "male") & ((df["Age"] <= 21) ), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 50) ), "NEW_SEX_CAT"] = "seniormale"

# Devamına bakamadım video error veriyor



#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
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

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

# Cabin_bool adlı bir değişken oluşturduğumuz için Cabin değişkenini dropladık
df.drop("CABIN", inplace=True, axis=1)

# Name üzerinden yeni değişkenler oluşturduk zaten, ticket ve bu 2 sini dropladık
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# New_title a göre age deki boşlukları doldurduk
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# Yaşa bağlı oluşturduğumuz değişkenleri tekrar oluşturduk. Çünkü yenile NaN'ları doldurduk
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# Tipi object olan ve eşsiz değişken syaısı 10 dan az olanları doldurduk
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

# 2 sınıflı değişkenleri aldık
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################

# Önce rare encoding yaptık ki indirgenebilecekleri indirgeyip o şekilde one-hot encoding'e sokalım
rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts() # Kontrol amaçlı yazdık

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2] # Zaten binary olanları dönüştürdüğümüzden 2 den fazla olanları seçtik

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

# Yeni oluşan değişkenlerde dağılımları yine çok olan olabilir. Gereksiz kolonlar oluşturmuş olabiliriz. O yüzden grab_col'u çağırıp rare_analyser'e sokacağız ve kolonların durumuna bakalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols) # Analizlere göre bazı bilgi taşımayan kolonlar çıktı (Ör NEW_FAMILY_SIZE_8 : 2 kolonunda survived'a etki eden sınıfın yüzdesi %0 imiş ve 6 taneymiş zaten)

# Bundan dolayı da bazı kolonları alıp target_mean'i 0.01 den düşük olanları kullanışsız olarak alalım
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True) # Silebiliriz de silmeyebiliriz de

#############################################
# 7. Standart Scaler
#############################################

# Bu problemde gerekli değil ama yine da yaptık. İhtiyaç olması durumunda böyle scaling yaparız
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # 0.80 çıktı. Yani %80

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # 0.70 Çıktı. Yani %70

# Yeni ürettiğimiz değişkenler ne alemde?

# Oluşturulan değişkenlerin modelimize nasıl bir etkide bulunduğuna baktık
# Sonuçta ise en yükarda new_name_count çıktı. Yani bu kolon en etkili kolonlardan

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