
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

mesajlar = [
    "Tebrikler ödül kazandınız",
    "Hemen tıklayın para kazanın",
    "Bedava hediye kazandınız",
    "Büyük indirim fırsatını kaçırmayın",
    "Ödül almak için bağlantıya tıklayın",
    "Ücretsiz para kazanma fırsatı",
    "Tebrikler büyük ödülü kazandınız",
    "Hediyenizi almak için hemen tıklayın",

    "Yarın okulda görüşürüz",
    "Toplantı saat kaçta başlayacak",
    "Bugün dersimiz var mı",
    "Akşam eve geleceğim",
    "Ödevini yapmayı unutma",
    "Yarın kahve içelim mi",
    "Toplantı yarın sabah yapılacak",
    "Bugün okuldan sonra görüşelim"
]

# 1 = Spam, 0 = Normal
etiketler = [
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
]

# Metinleri sayısal verilere dönüştürme
vec = CountVectorizer()
X = vec.fit_transform(mesajlar)

# Modeli eğitme
model = MultinomialNB()
model.fit(X, etiketler)

while True:
    mesaj = input("\nMesaj giriniz (çıkış: q): ")

    if mesaj.lower() == "q":
        break

    if not mesaj.strip():
        print("Lütfen mesaj giriniz.")
        continue

    yeni_veri = vec.transform([mesaj])
    tahmin = model.predict(yeni_veri)

    if tahmin[0] == 1:
        print("SPAM MESAJ")
    else:
        print("NORMAL MESAJ")
