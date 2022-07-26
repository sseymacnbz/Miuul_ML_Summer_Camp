print("hello world")
print("hello AI era")

##SAYILAR
9 #integer
9.2 #float

print(9)


###########################################
# Virtual Environment and Package Management
###########################################

# Sanal Ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create -n myenv

# Oluşturulan sanal ortamı aktif etme:
# conda activate my_new_env

#base ifadesi gitti ve my_new_env geldi

# Sanal ortamdan çıkıp base env'a dönme:
# conda deactivate

# Sanal ortamımızdaki yüklü paketleri görme:
# conda list
#Boş döndü çünkü hiçbir yükleme yapmadık

# conda ile paket yükleme işlemi:
# conda install numpy (numpy yerine indirilecek olan paketlerin ismi yazılır )

# Aynı anda birden fazla paket yükleme:
# conda install pandas numpy scipy (aralara boşluk bırakılarak paket isimleri yazılır)

# Paket silme :
# conda remove package_name

# Belirli bir versiyon numarasındaki paketi yükleme
# conda install numpy=1.20.1 (pipte 2 adet = kullanılmaktadır)

# Paket güncelleme:
# conda upgrade numpy

### pip ile paket yükleme işlemleri


# Paket yükleme:
# pip install package_name

# Versiyona göre paket yükleme:
# pip install pandas==1.2.1

# Projemizdeki paketleri bir başkasına göndermek isteyebiliriz. Bunu tek tek yapmak yerine
# yaml formatında conda ile oluşturup kolaylık sağlayabiliriz
# yaml formatı yml olarak da kullanılmaktadır

# conda env export > environment.yaml

# dir ile de dizindeki dosyaları listeledik

# Bir yaml dosyasını kendi environment'ımıza dahil etme
# conda env create -f environment.yaml

