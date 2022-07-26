##############################################
# VERİ YAPILARI (DATA STRUCTURES)
##############################################

##############################################
# Veri Yapılarına Giriş ve Hızlı Özet
##############################################

###
# Sayılar: integer
###
x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

###
# Sayılar: complex
###
x = 2j +1
type(x)

###
# String
###
x = "Hello AI era"
type(x)

###
# Boolean
###
True
False
type(True)

5==4

###
# Liste
###
x = ["btc", "eth", "xrp"]
type(x)

###
# Sözlük (dictionary)
###
x = {"name":"Peter", "Age":36}
type(x)

###
# Tuple (Demet)
###
x = ("python", "ml", "ds")
type(x)

###
# Set (Küme)
###
x = {"python","ml","ds"}
type(x)

# Not: Bu gördüğümüz liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak da bilinir.


###################################################################################


##############################################
# Sayılar (Numbers): int, float, complex
##############################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

###
# Tipleri Değiştirmek
###
int(b)
float(a)

int( a * b / 10)


###################################################################################

##############################################
# Karakter Dizileri (Strings)
##############################################

print("John")
"John"

name = "John"
print(name)


###
# Çok Satırlı Karakter Dizileri
###

long_str = """Veri Yapıları: Hızlı Özet"""

name[0]
name[3]

###
# Karakter Dizilerinde Slice İşlemi
###

name[0:2] #2 ye kadar git, 2 hariçtir
long_str[0:10]

###
# String İçinde Karakter Sorgulama
###
"veri" in long_str
"Veri" in long_str


##############################################
# String Metodları
##############################################

dir(int)
dir(str)# Stringlerle kullanılabilecek metodları listeledik

###
# len
###
name ="john"
type(name)
type(len)

len(name)#stringin boyut bilgisini döner

"""Not : Bir fonksiyon class yapısı içinde tanımlandıysa buna metod deriz
         Aksi halde de buna fonksiyon denir. Görevleri itibari ile de aynı şeylerdir"""

###
# upper() lower() : küçük-büyük dönüşümleri
###
"miuul".upper()
"MIuUL".lower()

#type(upper()) Bu bize hata verir. Bu bir metodtur

###
# replace : karakter değiştirir
###
hi = "Hello AI Era"
hi.replace("l","p")

###
# split : böler
###
hi = "Hello AI Era"
hi.split()

###
# strip : kırpar
###
" ofofo ".strip() #baştaki ve sondaki boşlukları kırptı
"ofofo".strip("o") #o ları kırptı

###
# capitalize : ilk harfi büyütür
###
"foo".capitalize()

###################################################################################

##############################################
# Liste (List)
##############################################

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir
# - Kapsayıcıdır

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]] #Kapsayıcıdırlar, birden fazla veri yapısını aynı anda tutarlar

not_nam[0]
not_nam[6]
not_nam[6][1]

notes[0]
notes[0] = 33
notes[0] # Listeler değiştirilebilirdirler


not_nam[0:4]


########
# Liste Metodları (List Methods)
#######

dir(list)

###
# len: built-in python fonksiyonu, boyut bilgisi verir
###
len(notes)
len(not_nam)

###
# append: eleman ekler
###
notes.append(100)
notes

###
# pop: indexe göre siler
###
notes.pop(0)
notes

###
# insert: indexe ekler
###

notes.insert(2, 15)
notes



###################################################################################

##############################################
# Sözlük (Dictionary)
##############################################

# - Değiştirilebilir
# - Sırasız. (Python 3.7'den sonra sıralı olma özelliği aldı)
# - Kapsayıcı


###
# key - value
###

dictionary = {"REG": "Regression",
              "CART": "Classification and Reg",
              "LOG": "Logistic Regression"}
dictionary["REG"]

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20]}
dictionary["LOG"][1]


###
# Key Sorgulama
###

"REG" in dictionary


###
# Key'e Göre Value Sorgulama
###

dictionary["REG"]
dictionary.get("REG")


###
# Value Değiştirmek
###

dictionary["REG"] = ["YSA", 10]

###
# Tüm Key'lere / Value'lara erişmek
###

dictionary.keys()
dictionary.values()


###
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
###

dictionary.items()


###
# Key-Value Değerini Güncelleme / Yeni Değer Ekleme
###

dictionary.update({"REG": 11})

dictionary.update({"RF": 10})



###################################################################################

##############################################
# Demet (Tuple)
##############################################

# - Değiştirilemez
# - Sıraldır
# - Kapsayıcıdır

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99 # Hata verir, çünkü değiştirilemez. t = list(t) diyip listeye çevirerek ancak değiştirme işlemi yapılabilir


###################################################################################

##############################################
# Set
##############################################

# - Değiştirilebilir
# - Sırasız + Eşsizdir
# - Kapsayıcıdır


###
# difference(): İki kümenin farkı
###

set1 = set([1, 2, 3])
set2 = set([1, 3, 5])

#set1'de olup set2'de olmayanlar
set1.difference(set2)

###
# symetric_difference(): İki kümede de birbirlerine göre olmayanlar
###
set1.symmetric_difference(set2)

###
# intersection(): İki kümenin kesişimi
###
set1.intersection(set2)

set1 & set2
set1 - set2


###
# union(): İki kümenin birleşimi
###
set1.union(set2)


###
# isdisjoint(): İki kümenin kesişimi boş mu?
###
set1.isdisjoint(set2)


###
# issubset():  Bir küme diğer kümenin alt kümesi mi ?
###

set1 = set([1, 2, 3])
set2 = set([1, 2, 3, 4, 5, 6])

set1.issubset(set2)


###
# issuperset():  Bir küme diğer kümeyi kapsıyor mu?
###

set2.issuperset(set1)


