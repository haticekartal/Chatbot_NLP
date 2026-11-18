
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Müşteri yorumları
yorumlar = [
    "Ürün çok güzel, çok beğendim",
    "Kargo hızlı geldi, teşekkür ederim",
    "Ürünün kalitesi harika",
    "Çok memnun kaldım",
    "Paketleme çok özenliydi",
    "Ürün beklentilerimi karşıladı",
    "Fiyatına göre çok kaliteli",
    "Satıcı çok ilgiliydi",
    "Ürün tam istediğim gibi",
    "Kesinlikle tavsiye ederim",

    "Ürün çok kötü, hiç beğenmedim",
    "Kargo çok geç geldi",
    "Ürünün kalitesi çok düşük",
    "Hiç memnun kalmadım",
    "Paketleme çok kötüydü",
    "Ürün beklentilerimi karşılamadı",
    "Verdiğim paraya değmedi",
    "Satıcı hiç ilgilenmedi",
    "Ürün bozuk geldi",
    "Kesinlikle tavsiye etmiyorum"
]

# Etiketler
# 1 = Olumlu, 0 = Olumsuz
etiketler = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]

# Verileri eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(
    yorumlar,
    etiketler,
    test_size=0.2,
    random_state=42,
    stratify=etiketler
)

# Metinleri sayısal verilere dönüştürme
vec = TfidfVectorizer()

X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)

# Naive Bayes modelini oluşturma
model = MultinomialNB()

# Modeli eğitme
model.fit(X_train_v, y_train)

# Test verileriyle tahmin yapma
tahminler = model.predict(X_test_v)

# Modelin başarı oranını hesaplama
basari = accuracy_score(y_test, tahminler)

print("Model başarı oranı:", basari)

print("\n--- MÜŞTERİ YORUM ANALİZİ ---")
print("Çıkmak için q yazınız.\n")

# Kullanıcıdan sürekli yorum alma
while True:

    yeni_yorum = input("Bir müşteri yorumu giriniz: ")

    # Programdan çıkış
    if yeni_yorum.lower() == "q":
        print("Program sonlandırıldı.")
        break

    # Boş yorum kontrolü
    if yeni_yorum.strip() == "":
        print("Lütfen bir yorum giriniz.\n")
        continue

    # Yorumu sayısal veriye dönüştürme
    yeni_yorum_v = vec.transform([yeni_yorum])

    # Tahmin yapma
    sonuc = model.predict(yeni_yorum_v)

    # Sonucu ekrana yazdırma
    if sonuc[0] == 1:
        print("Tahmin: OLUMLU YORUM\n")
    else:
        print("Tahmin: OLUMSUZ YORUM\n")
