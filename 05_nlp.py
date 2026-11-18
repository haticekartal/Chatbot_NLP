import re

metin = input("Bir metin giriniz: ")

# Küçük harfe çevirme
metin = metin.lower()

# Noktalama işaretlerini temizleme
metin = re.sub(r"[^\w\s]", "", metin)

# Gereksiz boşlukları temizleme
metin = " ".join(metin.split())

print("Temizlenmiş metin:", metin)
