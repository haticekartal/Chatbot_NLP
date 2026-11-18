
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

haberler = [
    "Futbol takımı maçı kazandı",
    "Basketbol turnuvası başladı",
    "Futbol liginde şampiyon belli oldu",
    "Takım yeni futbolcu transfer etti",

    "Yapay zeka teknolojisi gelişiyor",
    "Yeni bilgisayar modeli tanıtıldı",
    "Yazılım şirketi yeni uygulama geliştirdi",
    "Teknoloji dünyasında yapay zeka konuşuluyor",

    "Dolar kuru yükseldi",
    "Borsa bugün değer kaybetti",
    "Ekonomi piyasalarında hareketlilik yaşandı",
    "Bankalar faiz oranlarını açıkladı"
]

kategoriler = [
    "Spor", "Spor", "Spor", "Spor",
    "Teknoloji", "Teknoloji", "Teknoloji", "Teknoloji",
    "Ekonomi", "Ekonomi", "Ekonomi", "Ekonomi"
]

vec = TfidfVectorizer()
X = vec.fit_transform(haberler)

model = MultinomialNB()
model.fit(X, kategoriler)

while True:
    haber = input("\nHaber başlığı giriniz (q: çıkış): ")

    if haber.lower() == "q":
        break

    if not haber.strip():
        continue

    yeni_haber = vec.transform([haber])
    sonuc = model.predict(yeni_haber)

    print("Haber kategorisi:", sonuc[0])
