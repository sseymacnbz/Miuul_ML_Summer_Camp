########################################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
########################################################################
# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehensions

#####################################
# FONKSİYONLAR (FUNCTIONS)
#####################################

print("a", "b")
print("a", "b", sep="_")

help(print)


###
# Fonksiyon Tanımlama
###

def calculate(x):
    print(x * 2)


calculate(5)


def summer(arg1, arg2):
    print(arg1 + arg2)


summer(7, 8)
summer(arg1=8, arg2=7)


#####################################
# DOCSTRING
#####################################
# - Docstring, fonksiyonlara herkesin anlayabileceği bir şekilde bilgi notu ekleme yoludur

def summer(arg1, arg2):
    # """ yapıp enter'a bastıktan sonra doldurmamız gereken alanlar otomatik olarak geldi
    """
    Sum of two numbers

    :param arg1: int, float
        arguman1
    :param arg2: int, float
        argumanimiz2
    :return:
        int, float

    """

    print(arg1 + arg2)


# ?summer
# help(summer)
# Not: Arama kısmına docstrings yazıp arattırırsak ve tools altındaki alana gelirsek
# docstring'imizin ne formatta kullanılmak istediğimizi seçebiliriz.
# Default olanı reStructuredText'tir. En çok kullanılan format ise Google ve NumPy'dır
# Seçimimşz sonrası yukarıdaki docstring formatı değişebilmektedir ama mantık yine aynıdır
# NumPy formatındayken examples ve notes gibi alanlar da ekleyip docsting alanına örnek ve notlar ekleyebiliriz


#####################################
# FONKSİYONLARIN GÖVDE/STATEMENT BÖLÜMÜ
#####################################

def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# Girilen değerleri bir liste içinde saklayacak fonksiyon
list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 8)
add_element(180, 10)


#####################################
# ÖN TANIMLI ARGÜMANLAR/PARAMETRELER (DEFAULT/PARAMETERS/ARGUMENTS)
#####################################

def divide(a, b):
    print(a / b)


divide(1, 2)


def divide2(a, b=1):
    print(a / b)


divide2(10)
divide2(10, 2)


def say_hi(string="Merhaba"):
    print(string)
    print("Hi")
    print("Hello")


say_hi("mrb")
say_hi()


#####################################
# NE ZAMAN FONKSİYON YAZMA İHTİYACIMIZ OLUR ?
#####################################

# - Birbirini tekrar eden işlemler söz konusu olduğunda fonksiyon kullanılmalıdır


#####################################
# RETURN : Fonksiyon Çıktılarını Girdi Olarak Kullanmak
#####################################

def calculate(warm, moisture, charge):
    return (warm + moisture) / charge


calculate(98, 12, 78) * 10


def calculate(warm, moisture, charge):
    warm = warm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (warm + moisture) / charge

    return warm, moisture, charge, output


calculate(98, 12, 78)

warm, moisture, charge, output = calculate(98, 12, 78)

type(calculate(98, 12, 78))  # tuple'dır


#####################################
# FONKSİYON İÇERİSİNDEN FONKSİYON ÇAĞIRMAK
#####################################

def calculate(warm, moisture, charge):
    return int((warm + moisture) / charge)


calculate(98, 12, 78) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(warm, moisture, charge, p):
    a = calculate(warm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 12)





#####################################
# Local & Global Değişkenler/Variables
#####################################

list_store = [1, 2]
def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1, 9)
# c değişkeni burada local bir scope'tadır. list_store ise globaldedir





#####################################
# KOŞULLAR (CONDITIONS)
#####################################

###
# if
###

if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11
if number == 10:
    print("number is 10")

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)
number_check(10)

def number_check(number):
    if number == 10:
        print("number is 10")

    else:
        print("number is not 10")

number_check(12)
number_check(10)


def number_check(number):
    if number > 10:
        print("greater than 10")

    elif number < 10:
        print("less than 10")

    else:
        print("equal to 10")


number_check(12)
number_check(10)
number_check(6)


#####################################
# DÖNGÜLER (LOOPS)
#####################################

###
# for loop
###

students = ["John", "Mark", "Vanessa", "Mariam"]

for student in students:
    print(student)

for student in students:
    print(student.upper())


salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*20/100 + salary)) # %20 zam


def new_salary(salary, rate):
    return int(salary * rate/100 + salary)

for salary in salaries:
    print(new_salary(salary,10))


for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))

    else:
        print(new_salary(salary, 20))



#####################################
# Uygulama - Mülakat Sorusu
#####################################

# - Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

# Çift indexteki harfler büyük, tek indexteki harfler küçük olmalıdır

def alternating(string):
    new_string = ""

    # girilen string'in indexlerinde gezer
    for string_index in range(len(string)):#range bize belirtilen sayıya kadar değer üretir

        # index'in çift ve tek olmasına göre harf üzerinde büyültme ve küçültme işlemlerini uygular
        if string_index % 2 == 0:
            new_string += string[string_index].upper()

        else:
            new_string += string[string_index].lower()

    return new_string


string = "hi my name is john and i am learning python"
print(alternating(string))





#####################################
# Break & Continue & While
#####################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break # Döngüyü kırar
    print(salary)


for salary in salaries:
    if salary == 3000:
        continue # koşul sağlanırsa devam etme, iterasyona devam et
    print(salary)


number = 1
while number < 5:
    print(number)
    number += 1




#####################################
# Enumerate: Otomatik Counter/Indexer ile for loop
#####################################

#Bazen işlemler sırasında değerlere ait indexlere de ihtiyaç duyarız. Burada enumerate devreye girer
students = ["John", "Mark", "Vanessa", "Mariam"]

for student in students:
    print(student)


for index, student in enumerate(students):
    print(index, student)

for index, student in enumerate(students, 2):
    print(index, student)


A = []
B = []

for i, student in enumerate(students):

    if i%2 == 0:
        A.append(student)

    else:
        B.append(student)


print(A)
print(B)



#####################################
# Uygulama - Mülakat Sorusu Enumerate
#####################################

# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız
# Tek indexte yer alan öğrencileri bir listeye alınız
# Fakat bu iki liste tek bir liste olarak return olsun

def divide_students(students):
    groups = [[], []]

    for i, student in enumerate(students):
        if i%2 == 0:
            groups[0].append(student)

        else:
            groups[1].append(student)

    print(groups)
    return groups

divide_students(students)




#####################################
# Alternating Fonksiyonunu Enumerate ile Yazma
#####################################

def alternating_with_enumerate(string):
    new_string = ""

    for i, letter in enumerate(string):
        if i%2 == 0:
            new_string += letter.upper()

        else:
            new_string += letter.lower()

    print(new_string)


alternating_with_enumerate("hi my name is john")



#####################################
# Zip
#####################################

students = ["John", "Mark", "Vanessa", "Mariam"]
departments = ["mathematics", "statistics", "physics", "astronomy"]
ages = [23, 30, 26, 22]

# Amacımız 3 listenin elemanlarını eşlemek olsun

list(zip(students, departments, ages))


#####################################
# Lambda, Map, Filter, Reduce
#####################################

# - lambda kullan at anlamında bir fonksiyon tanımlama aracıdır
def summer(a, b):
    return a+b

new_sum = lambda a, b: a + b
new_sum(4, 5)

# - map, bizi döngü yazmaktan kurtarır
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

# map der ki: Bana bir fonksiyon ver, bu fonksiyonu da uygulamak istediğim iteratif bir nesne ver
# böylece for yazmaktan kurtul
list(map(new_salary, salaries))


# lambda ve map'in beraber kullanımı

list(map(lambda x: x * 20 / 100 + x, salaries)) # Yukarıda yazdığımız satırları 1 satıra indirdik


# - filter: filtreleme işlemleri için kullanılır

list_store = [1,2,3,4,5,6,7,8,9,10]
list(filter(lambda x: x%2 == 0, list_store))


# - reduce
from functools import reduce

list_store = [1,2,3,4]
reduce(lambda a, b: a+b, list_store)






#####################################
# COMPREHENSIONS
#####################################


##########################
# List Comprehensions
##########################

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))


null_list = []
for salary in salaries:
    null_list.append(new_salary(salary))


for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))

    else:
        null_list.append(new_salary(salary * 2))


# - Bu yaptıklarımızı list comprehensions ile yaparsak da;

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]
# Tek bir satırda olur. Şimdi adım adım nasıl ilerledik bakalım

# Maaşlar listesindeki her maaşı 2 ile çarpalım
[salary * 2 for salary in salaries] # Sonuç liste ama kaydetmedik bir yere

# Maaşı 3000 den az olanları 2 ile çarpalım
[salary * 2 for salary in salaries if salary < 3000]

# Comprehension kullanırken tek bir if ifadesi varsa for solda, if sağda kalır
# Bir else ifadesi de kullanılacaksa if else yapısı solda, for yapısı sağda kalır !!!
[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

# Elimizde var olan bir fonksiyonu da işin içine katarsak;
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]


students = ["John", "Mark", "Vanessa", "Mariam"] # Öğrenci listesi
students_no = ["John", "Vanessa"] # İstenmeyen öğrenci listesi

#İstenmeyen öğrencilerin isimlerini küçük, ötekilerinkinin ismini ise küçük yazmak isteyelim
[student.lower() if student in students_no else student.upper() for student in students]

#Yukardaki yapıyı "not in" yapısı kullanarak yazarsak;
[student.upper() if student not in students_no else student.lower() for student in students]



##########################
# Dict Comprehensions
##########################

dictionary = {"a":1,
              "b":2,
              "c":3,
              "d":4}

dictionary.keys()
dictionary.values()
dictionary.items()

# Key'lere dokunmadan her bir value'nun karesini alalım
{k: v ** 2 for (k, v) in dictionary.items()} # k:keyleri, v:valueları temsil etsin

# Keylere müdahele edelim
{k+"x": v for (k, v) in dictionary.items()}

# İkisine de müdahale edelim
{k.upper(): v*2 for (k, v) in dictionary.items()}


##########################
# Uygulama - Dict Comprehensions Mülakat Sorusu
##########################

# Amaç : Çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir
# Key'ler orjinal değerler, value'lar ise değiştirilmiş(karesi alınan) değerler olmalıdır
numbers = range(10)

# - Normal yoldan yaparsak;

new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

# Dict Comprehension ile yaparsak;
{n: n**2 for n in numbers if n%2 == 0}


##########################
# List & Dict Comprehension Uygulamalar
##########################

# 1- Bir Veri Setindeki Değişken İsimlerini Değiştirmek

# before:
# ["total", "speeding", "not_distracted", "no_previous"]

# after:
# ["TOTAL", "SPEEDING", "NOT_DISTRACTED", "NO_PREVIOUS"]

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

# Klasik yöntemle çözersek;

A = []
for col in df.columns:
    A.append(col.upper())

df.columns = A # Kolon isimlerini büyüttük

# Comprehension ile yaparsak;
df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]


# 2- Kolonun isminde "INS" olanların başına "FLAG", diğerlerine "NO_FLAG" ekleyelim

[col for col in df.columns if "INS" in col] # İçerisinde "INS" olanları getir
["FLAG_"+col for col in df.columns if "INS" in col] # İçerisinde "INS" olanların başına flag ekle

["FLAG_"+col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_"+col if "INS" in col else "NO_FLAG_" + col for col in df.columns] #Kalıcı olarak isimleri değiştirdik


# 3- Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak
#   Bunu sadece sayısal değişkenler için uygulayacağız

# Output:
# {"total": ["mean","min","max","var"],
# "speeding": ["mean", ]...

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

#Veri setindeki sayısal kolonları seçme
num_cols = [col for col in df.columns if df[col].dtype != "O"] # dtype'ı object olmayanlar alınacaktır

soz = {}
agg_list = ["mean", "min", "max", "var"]

for col in num_cols:
    soz[col] = agg_list

# dict comphrehension ile;
new_dict = {col: agg_list for col in num_cols}
df[num_cols].agg(new_dict)

