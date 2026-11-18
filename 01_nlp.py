
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

# Yorumların etiketleri
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
    random_state=42
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

# Modelin başarı oranı
basari = accuracy_score(y_test, tahminler)

print("Model başarı oranı:", basari)

# Yeni bir müşteri yorumu
yeni_yorum = ["kargo çok geç geldi, memnun kalmadım"]

# Yeni yorumu sayısal veriye dönüştürme
yeni_yorum_v = vec.transform(yeni_yorum)

# Yeni yorumu sınıflandırma
sonuc = model.predict(yeni_yorum_v)

if sonuc[0] == 1:
    print("Yorum: Olumlu")
else:
    print("Yorum: Olumsuz")
