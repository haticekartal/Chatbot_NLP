
from collections import Counter

metin = input("Bir metin giriniz: ")

kelimeler = metin.lower().split()

sayac = Counter(kelimeler)

print("Kelime sayısı:", len(kelimeler))
print("Farklı kelime sayısı:", len(sayac))

print("\nEn sık kullanılan kelimeler:")

for kelime, adet in sayac.most_common(5):
    print(kelime, ":", adet)
