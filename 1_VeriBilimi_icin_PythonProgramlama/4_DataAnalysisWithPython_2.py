#########################################################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
#########################################################################

######################################
# MATPLOTLIB
######################################
# Veri görselleştirmenin atasıdır. Low-level bir veri görselleştirme aracıdır. Yani daha fazla çabayla veri görselleştirme işlemi gerçekleştirilmektedir

# Kategorik değişken: sütun grafiği. countplot or barplot
# Sayısal değişken: hist, boxplot.
# PowerBI, Tableu gibi araçlar veri görselleştirmeye daha uygundur.
# Python küçük çaplı görselleştirmelerimiz için kullanıma uygundur.
# Bir işyerinde veriler genelde veri tabanında olacağından Python daha az uygunluk gösterir

##############
# Kategorik Değişken Görselleştirme
##############
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg") #Çıktı ekranı yanıt vermiyor şeklinde dönüt verdiği için bu yazıldı
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()


##############
# Sayısal Değişken Görselleştirme
##############

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

##############
# Matplotlib'in Özellikleri
##############
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#####
# plot
#####

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o") # Noktaları işaretleyerek grafik oluşturur
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()


#####
# marker : işaretçi
#####
y = np.array([13, 28, 11, 100])
plt.plot(y, marker = "o") # belirttiğimiz noktaları işaretler
plt.show()

markers = ["o", "*", ".", ",", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h"] #Bunlar da marker olarak kullanılmaktadır


#####
# line : çizgi
#####
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashed", color="r")
plt.show()


#####
# multiple lines
#####

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()


#####
# labels
#####

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
plt.title("Bu ana başlık")
plt.xlabel("X ekseni")
plt.ylabel("Y ekseni")
plt.grid() # Izgara
plt.show()

#####
# subplots
#####
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1) # 1 satır, 2 sütunluk grafik oluştur. 1. grafiğe bunu yerleştir
plt.title("1")
plt.plot(x, y)

x = np.array([8,8,9,9,10,15,11,15,12,15,])
y = np.array([24,20,26,27,280,29,30,30,30,30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)



######################################
# SEABORN
######################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data = df)
plt.show()

# matplotlib ile;
df["sex"].value_counts().plot(kind="bar")
plt.show()


sns.boxplot(x=df["total_bill"])
plt.show()

# pandas ile hist çizdirme
df["total_bill"].hist()
plt.show()


# ********************************************************************** #

#########################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#########################################################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Target Variable)
# 4. Hedef Değişken Analizi (Analysis of Target Varibale)
# 5. Korelasyon Analizi (Analysis of Correlation)

# Amaç: Hızlı bir şekilde genel fonksiyonlar ile elimize gelen verileri analiz etmektir

#########################################
# 1. Genel Resim
#########################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

# Elimize bir veri ilk kez geldiğinde uygulanabilecek olan fonksyionlar
df.head()
df.tail()
df.shape
df.info()
df.columns
df.tail()
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
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

check_df(df)


#########################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#########################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique() # Toplamda kaç unique değer var? number unique

# Öyle bir şey yapmalıyız ki bize olası tüm kategorik değişkenleri versin

# titanic veri setinde 'survived' veya 'pclass' gibi integer tanımlı değişkenler her ne kadar integer gibi gözükse de aslında kategorik değişkenlerdir
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]] #Tüm kolonlarda gez, eğer colonun dtype'ı category, object veya bool ise al
# df["sex"].dtypes => çıktı: dtype('O')     str(df["sex"].dtypes) => çıktı: 'object'. Bu sebeple yukarda str içine aldık

# Şimdi de integer görünümlü kategorik değişkenleri yakalayalım
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["int64", "float64"]] # Tipi integer veya float olanlardan unique değeri belirli bir sayıdan az olanları aldık

# Veri setinde isimler diye bir değişken olsayıd 891 adet unique içerikli bir kategorik değişkeni de işin içine katmış olurduk
# Lakin bu bir kategorik değişken değildir. Bu gibi değişkenlere kardinalitesi yüksek değişkenler denir.
# Açıklanabilir değer taşımayan değişkenlerdir. Bu değişkenleri de yakalayabilmeliyiz
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

# Artık tüm kategorik kolonları bir araya getirebiliriz
cat_cols = cat_cols + num_but_cat

# Şimdilik cat_but car boş geldi ama doldu olduğunda da şöyle yapmalıydık
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols]

df[cat_cols].nunique()

#100 * df["survived"].value_counts() / len(df) # İlgili kategorik değişkenin içeriğindeki değerlerin oranlarını verir
def cat_summary (dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("######################################################")
    # Daha iyi bir görünüm olması için dataframe e çevirdik


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("######################################################")

    if plot:
        sns.countplot(x=dataframe[col], data =dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True) # Bu fonksiyona bir de görselleştirme işlemi ekledik. Lakin buraya bir bool değişken gelseydi ona bir görselleştirme işlemi yapamaz hata verirdi. Çünkü bool verilerde sayma işlemi yapamamaktadır

# Şu şekilde bir düzeltme yaparsak;

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int) # Böylece True-False'ları 1-0'lara çevirdik ve hatadan kurtulmuş olduk
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df,col, plot=True)


#########################################
# 3. Sayısal Değişken Analizi
#########################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

df[["age","fare"]].describe().T

# Veri setinden sayısal değişkenleri nasıl seçeriz?
num_cols = [col for col in df.columns if str(df[col].dtypes) in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols] # cat_cols da olmayanları getirdik. Çünkü bazı sayısal görünümlü veriler aslında kategorikti

def num_summery(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summery(df, "age")

for col in num_cols:
    num_summery(df, col)

# Summary fonksiyonumuza bir de plot özelliği ekleyelim
def num_summery(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

num_summery(df, "age", True)

#Tüm sayısal veriler için;
for col in num_cols:
    num_summery(df, col, plot=True)



#########################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#########################################

# Öyle bir fonksiyon yazmalıyız ki bize kategorik değişken listesini, nümerik değilken listesini,
# kategorik ve kardinal listesini versin

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

#Artık aşağıdaki fonksiyonları daha rahat kullanabiliriz
def cat_summary (dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("######################################################")
    # Daha iyi bir görünüm olması için dataframe e çevirdik

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

for col in num_cols:
    num_summary(df, col, plot=True)

# BONUS: bool tipindeki verileri int'e çevirip cat_summary fonksiyonunu plot özelliği olacak şekilde baştan yazalım
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("######################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data =dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)



#########################################
# 4. Hedef Değişken Analizi
#########################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)



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

# Hedef değişkenimiz olan "survived" değişkenini analiz etmek istiyoruz
# Hayatta kalanlar neden hayatta kalıyorlar, hayatta kalma sebepleri nelerdir

###################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
###################

df.groupby("sex")["survived"].mean() # survived'ı int'e çevirdiğimiz için işlemi gerçekleştirebildik
# Kadınların erkeklerden daha fazla hayatta kalma oranı vardır

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "pclass") # First class yolcular daha çok hayatta kalmış

for col in cat_cols:
    print("")
    target_summary_with_cat(df, "survived", col)



###################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
###################

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"}) # Yukardakiyle aynı çıktı

# Bunu bir fonksiyona çevirirsek;

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    print(type(df[col]))
    target_summary_with_num(df, "survived", col)


#########################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#########################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("VeriBilimi_icin_PythonProgramlama/datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1] # İstemediğimiz değişkenler vardı onları çıkardık
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]

corr = df[num_cols].corr()
# Korelasyonu birbirine çok yakın olan 2 değer varsa bunlarn birbirine çok benzediğini kabul edip bunlardan birini elimine etmek isteriz

sns.set(rc={"figure.figsize":(12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()



###############################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
###############################
# Her projede bunu yapmak zorunda değiliz.

cor_matrix = df.corr().abs() # Bizim için şu an negatif veya pozitifliği değil, korelasyonun ne kadar yüksek olduğu. O yüzden de korelsyonların mutlak değerlerini aldık

# Korelasyon matrisinde kesişimler ve altta kalan kısmı Na yaparak matrisi sadeleştirelim
upper_triangle = cor_matrix.where(np.triu(np.zeros(cor_matrix.shape), k=1).astype(np.bool)) # oluşturulan korelasyon matrisi boyutlarında ve birlerden oluşan bir numpy array'i oluşturuldu. Bu array'i bool'a çeviriyoruz. Daha sonra köşegen elemanlardan kurtulduk.

# Şimdi de sütunlarda %90 dan büyük olan korelasyon değeri varsa o kolonu silelim
drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.90)] # any ifadesi ile kolon içindeki tüm değerleri gözden geçirebildik
df.drop(drop_list, axis=1)

# Bu işlemi bir fonksyion haline getirelim
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, plot=True)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)
# Yukarıda önce direkt df'e uyguladık. Çıkan grafikte fazlaca yüksek korelasyonlar gözükmekteydi. Bu yüksek korelasyonları silinceelimize daha sade bir görsel geldi


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)


# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)



