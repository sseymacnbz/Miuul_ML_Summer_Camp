############################################################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
############################################################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)


#####################################################
# NUMPY
#####################################################
# Python dünyasını temel programlama dünyasından veri analitiği dünyasına açan kütüphanedir
# Neden NumPy?:
# NumPy arrayleri listelere göre daha hızlı işlemlerde bulunur. Bunu da fix bir veri tipindeki verileri tutarak sağlar
# Vekötrel seviyeden(yüksek seviyeden) işlemler yapmamızı sağlar. Böylece az kodla çok iş yapmamızı sağlar
# Ex:

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i]) # a * b

# Bu işlemi bir de numpy ile yaparsak:

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b # a * b


#####################################################
# NumPy Array'i Oluşturmak (Creating NumPy Arrays)
#####################################################

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0, 10, size = 10) # 0-10 arasında rastgele 10 sayı üretir
np.random.normal(10, 4, (3, 4)) # Ortalaması 10 ve std'si 4 olan 3'e 4'lük bir array


#####################################################
# NumPy Array Özellikleri (Attributes of NumPy Array)
#####################################################

a = np.random.randint(10, size=5)
a.ndim # Boyut sayısı
a.shape # Boyut bilgisi
a.size # Toplam eleman sayısı
a.dtype # İçerisinde tuttuğu veri tipi


#####################################################
# Yeniden Şekillendirme (Reshaping)
#####################################################

a = np.random.randint(1, 10, size=9) # Bu arrayi 3'e 3'lük yapmak istersek
ar = a.reshape(3,3)

# Not: 10 elemanlı array'i 3'e 3 yapamayız.


#####################################################
# Index Seçimi (Index Selection)
#####################################################

a = np.random.randint(10, size=10)

a[0] # 0. index
a[0:5] # Slicing

m = np.random.randint(10, size=(3, 5))
m[0, 0] # [Satır, Sütun]
m[0:2, 0:3]

m[2, 3] = 2.9
# Burada 2,3 değeri 2.9 olmaz. np arrayler fix typelıdır. O yüzden [2,3] = 2 olur

m[:, 0]
m[1, :]



#####################################################
# Fancy Index
#####################################################

v = np.arange(0, 30, 3) # 0'dan 30'a kadar(30 hariç) 3 er artarak

v[1]
v[4]

catch = [1, 2, 3]
v[catch] # Fancy index sayesinde tek tek çağırmak yerine bir liste yardımıyla istediğimiz indexleri çağırdık



#####################################################
# NumPy'da Koşullu İşlemler (Conditions on NumPy)
#####################################################
# Amacımız, array'deki 3 ten küçük değerlere ulaşmak olsun
v = np.array([1, 2, 3, 4, 5])

##
# Klasik yol
##
ab = []
for i in v:
    if i < 3:
        ab.append(i)

##
# NumPy ile
##
v < 3 # Array'in tüm elemanlarında kontrol yapıp True veya False döner

v[v < 3] #Yukarıdaki 4,5 satır yerine tek bir satırda hallettik
# Arka mantıkta yine Fancy işlemlerinin gerçekleşmesi vardır.



#####################################################
# Matematiksel İşlemler (Mathematical Operations)
#####################################################

v = np.aarray([1, 2, 3, 4, 5])

v/5 # İçerideki tüm elemanlara bunu uygular
v*5/10
v ** 2

# Bu işlemleri metodlar ile de gerçekleştirebiliriz

np.subtract(v, 1) # Çıkartma
np.add(v, 1) # Ekleme
np.mean(v) # Ortalam
np.sum(v) # Toplama
np.min(v) # min değer
np.max(v) # max değer
np.var(v) # varyans

###
# NumPy ile 2 Bilinmeyenli Denklem Çözümü
###

# 5 * x0 + x1 = 12
#x0 + 3 * x1 = 10

a = np.array([[5, 1], [1, 3]]) # Katsayılar
b = np.array([12, 10]) # Sonuçlar

np.linalg.solve(a, b)



#####################################################
# PANDAS
#####################################################

# Pandas Series
# Veri okuma (Readig Data)
# Veriye HIzlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#  Toplulaştırmma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri


#####################################################
# Pandas Seires
#####################################################
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)

s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head()

#####################################################
# Veri Okuma (Reading  Data)
#####################################################

df = pd.read_csv("VeriBilimi_icin_PythonProgramlama/datasets/advertising.csv")
df.head()


#####################################################
# Veriye Hızlı Bakış (Quick Look at Data)
#####################################################
import seaborn as sns

df =  sns.load_dataset("titanic")
df.head()
df.tail()

df.shape # boyut bilgisi
df.info() # df'in değişkenleri hakkında bilgiler
df.columns
df.index
df.describe() # df'e ait betimsel bilgiler

df.isnull().values().any() # en az 1 tane bile olsa null değer var mı ?
df.isnull().sum() # df.isnull() True-False döner. T=1, F=0 demektir. Biz buna bir de sum() eklersek 1'leri toplar ve böylece toplam null dğerlerini görmüş oluruz
df["class"].value_counts() # Değişkene ait değerlerden kaçar tane olduğunu döner


#####################################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#####################################################

df[0:13]
df.drop(0,axis=0) # axis=0 satır demektir. 0. indexi siler
df.drop([1, 3, 4, 7], axis=0)

# Bu silme işlemleri kalıcı değildir. df = df.drop() şeklinde atama yappılabilir
# veya inplace = True işlemi  yapılabilir

###
# Değişkeni Indexe Çevirme
###

# Yaş değşkenini index'e çevirelim
df.index = df["age"]
df.drop("age" ,axis=1, inplace=True)

###
# Indexi Değişkene Çevirme
###
# 1. Yol
df["age"] = df.index
# "age" adlı kolonu silmiştik. Bu kolonu bulamazsa kendisi oluşturur.


# 2. Yol
df = df.reset_index()  # Indexte yer alan değerleri siler ve yeniden 0'dan itibaren numaralandırma yapar


#####################################################
# Değişkenler Üzerinde İşlemler
#####################################################

pd.set_option("display.max_columns", None) # Gösterilecek maximmum kolon sayısı olmasın . Böylece konsolda '...' yerine tüm  veriler gelir
df = sns.load_dataset("titanic")
df.head()

"age" in df # Değişken veri setinde var mı?

df[["age"]].head() # Tek bir [] ile series elde ederiz. 2 tane [] ile dataframe elde ederiz. Yani [] içine bir  liste göndermiş olmaktayız
df["age2"] = df["age"] ** 2

df.loc[:, df.columns.str.contains("age")] # columns'lar için bir str işlemi uygulayavağımızı söyledik ve içinde "age" olan  kolnları getirdi
# age olmayanları isteseydik ~df.columns.str.contains("age") şeklinde yazmamız yeterliydi


#####################################################
# loc & iloc
#####################################################
df = sns.load_dataset("titanic")

# iloc : integer based selection
df.iloc[0:3]

# loc: label based selection
df.loc[0:3]

# iloc ile yaptığımızda 3. index hariç 0,1,2. indexler alındı.
# loc ile yaptığımızda ise 4 adet veri döndürdü. 3. dahil olmuş oldu

# 0-3 satırlarını ve bunları da sadece "age" değişkeninden almak isteyelim
df.iloc[0:3, "age"] # hata verir. iloc, integer based dir.
df.loc[0:3,  "age"] # hata vermez. İstediğimiz şekilde çalışır

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names] # fancy index kavramı ile bir liste yardımıyla seçim işlemi gerçekleştirdik


#####################################################
# Koşullu Seçim (Connditional  Selection)
#####################################################

df[df["age"]> 50].head() # Yaşı 50'den büyük olanlar
df[df["age"]>50]["age"].count() # Yaşı 50'den büyük kaç kişi var

df[df["age"]>50]["class"].value_counts()
df.loc[df["age"] > 50, ["age", "class"]].head() # Yaşı 50'den büyük olanların class'ları nedir?

df[(df["age"] > 50) & (df["sex"] == "male")][["age", "class"]] # Yaşı 50'den büyük olan erkekler
# Koşullar muhakkkak parantez içerisine alınması gerekmektedir

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]] # Yukarıdakiyle ayı çıktıyı verir

# embark_town değşkeni "Southampton"  ya da "Cherbourg" olanları getirelim

df[(df["embark_town"] == "Southampton") | (df["embark_town"] == "Cherbourg")]
# Not: Koşullardan birini () içine almadığım için empty dataframe döndü. Bu kısma dikkat et



#####################################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#####################################################

# Toplulaştırma => Bizeözet istatistikler veren fonksiyonlardır
# Gruplama =>

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

# Kadınların ve erkeklerin yaş ortalamalarına ulaşmak isteyelim

df.groupby("sex")["age"].mean() # Cinsiyet groupby'ında yaşın ortalaması alındı

# Sadece ortalama değil toplamını da almak isteyelim
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean","sum"]}) # Cınsiyete göre kırdık

df.groupby("sex").agg({"age": ["mean","sum"], # Kadın ve erkeklerin hayatta kalma oranlarıan baktık
                       "survived":"mean"})

df.groupby(["sex","embark_town"]).agg({"age": ["mean","sum"],
                       "survived":"mean"}) # Veri önce kadın-erkek olarak sonra gemiye bindikleri lokasyona göre bölündü

df.groupby(["sex","embark_town","class"]).agg({"age": ["mean","sum"],
                       "survived":"mean"}) # Bazı survived ortalamaları 0 olmakkta. Burada elimizde bir frekans  olmadığı için 100 kişinin hepsi mi ölmüş yoksa 1 kişi vardı o mu ölmüş anlayamayız

df.groupby(["sex","embark_town","class"]).agg({
    "age": ["mean"],
    "survived":"mean",
    "sex":"count"})

#####################################################
# Pivot Table
#####################################################
pd.set_option("display.max_columns", None)
# Yaş ve gemiye binme lokasyonlar için pivot tablo oluşturalım ve survived bilgilerine ulaşalım

df.pivot_table("survived","sex","embarked") # 1. argüman..: Kesişimlerde neyi görmek istiyorsan onu gir. 2. argüman..: indexte yani satırda hangi değişkeni görmek istiyorsun. 3.argüman..: Sütunda hangi değişkeni görmek istiyorsun
# pivot table da ön tanımlıı değer mean dir. Girilen değerlerin kesişiminin ortalamasını alır

df.pivot_table("survived","sex","embarked", aggfunc="std") # standart sapma hesapladı

df.pivot_table("survived","sex",["embarked", "class"]) # Sütunlarda 2 index varken satırlarda 1 indexlik bilgi vardır.

# Yaş değişkenini kategorik bir hale getirip yaşın survived üzerine etkisini inceleyelim
# Yani yaş kırılımında hayatta kalma durumlarını inceleyelim
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90]) # cut ve qcut fonksiyonları elimizdeki sayısal verileri kategorik verilere çevirmemizi sağlar. Neyi böleceğimizi ve nerden böleceğimizi parametre olarak alır
                         # Bu örnekte 0-10 arası bir grup, 10-18 arası bir grup ... şeklinde cut işlemi gerçekleştirilir
                         # Sayısal değişkeni hangi kategorilere böleceğimizi biliyorsak cut(), çeyreklik değerlere bölmek istiyorsak qcut() fonksiyonu kullanılır
                         # Ör: 0-10 arası yaşa çocuk, 10-25 genç vs gibi böleceksek cut() kullanırız

df.head()   # Kesişimlerdeki , satırlardaki,  sütunlardaki    değişkenleri belirledik
df.pivot_table("survived",     "sex",         "new_age")

df.pivot_table("survived","sex",["new_age", "class"])

pd.set_option("display.width", 500) # Çıktılar uzun olunca \ ile alta kayıp durmaktaydı. Bu sayede çıktıları yan yana görebilmekteyiz


"""
pd.cut(df["age"], [0, 10, 18, 25, 40, 90],
                       labels=["cocuk", "genc", "yetiskin", "orta_yasli", "yasli"])
# labels etiketiyle aralıkları isimlendirebilriz.
"""


#####################################################
# Apply ve Lambda
#####################################################
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

# Apply satır ya da sütunlarda otomatik olarak fonksiyon çalıştırma imkanı sunar
# Lambda ise fonksiyon tanımlama şeklidir. Kod akışı sırasında kullan at bir şekilde fonksiyon tanımlamamızı sağlar

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

# Veri seti içerisindeki yaş değişkenlerini 10'a bölmek isteyelim
# Klasik yol

# df["age"]/10.head() hatalı bir kullanım
(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

# veya
for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

# Kısa yol

df[["age","age2","age3"]].apply(lambda x: x/10).head() # Kolonları seç, uygula(apply) şu fonksiyonu(lambda) => kendisine girilen ifadelerin 10'a bölümünü yapan bir fonksiyon

# İşi daha programatik yapmak istersek
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10) # df.loc[Tüm satırları seç , kolonlarda string işlemi yapıcaz. "age" içerenleri al].apply ...
# apply bize değişkenleri gezmemizi sağladı

# Bir fonksiyon olsun, öyle ki uygulandığı dataframe'deki değerleri standartlaşsın. Yani bir normalleştirme işlemi yapalım
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()) # High level bir satır yazık. seçilen age değerinden o kolondaki tüm age değerlerinin ortalamasını alarak işlem yapar Ör: 22 - (age kolonunun ortalaması)

# Yukarıdaki ifadeyi normal bir fonksiyon ile yazarsak

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

# Bunu yaş kolonlarına atamak istersek;

df.loc[["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
# veya
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)


#####################################################
# Birleştirme (Join) İşlemleri
#####################################################
import pandas as pd
import numpy as np
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])
df2 = df1 + 99

# Alt alta birleştirme
pd.concat([df1, df2])

# Yukarıda indexleri olduğu gibi tutar. Indexleri düzeltmek istersek.
pd.concat(["df1", "df2"], ignore_index=True)

# Yan yana birleştirmek istersek
pd.concat([df1, df2], axis=1) # default 0 olduğundan alt alta birleştirmişti

###
# Merge ile birleştirme işlemi
###

df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2) # Hangi değişkene göre birleştirme işlemi yapacağını vermediğimiz halde işlem gerçekleşti. Bunu kendimiz belirtmek istersek 'on' argümanının kullanırız
pd.merge(df1, df2, on = "employees")

# Her çalışanın müdür bilgisini almak isteyelim
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                    "manager" : ["Caner", "Mustafa", "Berkcan"]})

pd.merge(df3, df4) # Otomatik olarak group ortak değişkeni ile birleştirme yapar