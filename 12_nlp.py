
# UYGULAMA: YAPAY ZEKA DESTEKLI BANKACILIK CHATBOTU
# TF-IDF + Naive Bayes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. EGITIM VERILERI

sorular = [
    # Bakiye sorgulama
    "Bakiyem ne kadar",
    "Hesabımda kaç TL var",
    "Param ne kadar",
    "Hesap bakiyemi göster",
    "Ne kadar param kaldı",
    "Hesabımdaki parayı öğrenmek istiyorum",

    # Hesap hareketleri
    "Son işlemlerimi göster",
    "Hesap hareketlerim neler",
    "En son nereye para harcadım",
    "Geçmiş işlemlerim",
    "Hesap geçmişimi göster",
    "Son harcamalarım neler",

    # Para transferi
    "Para göndermek istiyorum",
    "Havale yapmak istiyorum",
    "Başka hesaba para aktar",
    "Para transferi yap",
    "Birine para göndereceğim",
    "EFT yapmak istiyorum",

    # Kredi kartı
    "Kredi kartı limitim ne kadar",
    "Kart limitimi öğrenmek istiyorum",
    "Kredi kartımın limiti nedir",
    "Kullanılabilir kart limitim",
    "Kartımda ne kadar limit var",
    "Kredi kartı bilgilerini göster",

    # Kayıp kart
    "Kartımı kaybettim",
    "Kartım çalındı",
    "Kartımı bloke etmek istiyorum",
    "Kayıp kart bildirimi",
    "Kartımı kapatmak istiyorum",
    "Kredi kartım kayboldu"
]

# 2. ETIKETLER

etiketler = (
    ["bakiye"] * 6 +
    ["hareketler"] * 6 +
    ["transfer"] * 6 +
    ["kredi_karti"] * 6 +
    ["kayip_kart"] * 6
)

# 3. METINLERI SAYISAL VERILERE DONUSTURME

vec = TfidfVectorizer(ngram_range=(1, 2))

X = vec.fit_transform(sorular)

# 4. MODELI EGITME

model = MultinomialNB(alpha=0.5)
model.fit(X, etiketler)

# 5. ORNEK BANKA BILGILERI

bakiye = 12500

kart_limiti = 20000

kart_bloke = False

hareketler = [
    "Market alışverişi: -350 TL",
    "Elektrik faturası: -750 TL",
    "Maaş ödemesi: +25000 TL",
    "İnternet faturası: -400 TL"
]

# 6. CHATBOT BASLANGICI

print("================================")
print("       DEMOBANK CHATBOT")
print("================================")

isim = input("Adınız nedir? ").strip()

if isim == "":
    isim = "Müşteri"

print("\nMerhaba", isim)
print("DemoBank'a hoş geldiniz!")

print("\nYapabileceğiniz işlemler:")
print("- Bakiye sorgulama")
print("- Hesap hareketleri")
print("- Para transferi")
print("- Kredi kartı limiti")
print("- Kayıp kart bildirimi")

print("\nÇıkmak için q yazınız.\n")

# 7. SOHBET DONGUSU

while True:

    mesaj = input(isim + ": ").strip()

    if mesaj.lower() == "q":
        print("Bot: İyi günler", isim)
        break

    if mesaj == "":
        print("Bot: Lütfen bir mesaj yazınız.")
        continue

    # Mesajı sayısal verilere dönüştürme
    yeni_mesaj = vec.transform([mesaj])

    # Tanınan kelime kontrolü
    if yeni_mesaj.nnz == 0:
        print("Bot: İsteğinizi anlayamadım.")
        print("Bot: Bankacılık işlemleriyle ilgili soru sorabilirsiniz.\n")
        continue

    # Kullanıcının niyetini tahmin etme
    niyet = model.predict(yeni_mesaj)[0]

    print("\nBot:")

    # BAKIYE SORGULAMA
    if niyet == "bakiye":

        print("Güncel bakiyeniz:", bakiye, "TL")

    # HESAP HAREKETLERI
    elif niyet == "hareketler":

        print("Son hesap hareketleriniz:")

        for hareket in hareketler:
            print("-", hareket)

    # PARA TRANSFERI
    elif niyet == "transfer":

        print("Para transferi işlemi başlatıldı.")

        alici = input("Alıcı adı: ")

        try:
            tutar = float(input("Gönderilecek tutar: "))

            if tutar <= 0:
                print("Bot: Tutar sıfırdan büyük olmalıdır.")

            elif tutar > bakiye:
                print("Bot: Yetersiz bakiye!")

            else:
                onay = input("Transferi onaylıyor musunuz? (e/h): ")

                if onay.lower() == "e":

                    bakiye = bakiye - tutar

                    hareketler.insert(
                        0,
                        alici + " kişisine transfer: -" + str(tutar) + " TL"
                    )

                    print("Bot: Demo transfer gerçekleştirildi.")
                    print("Kalan bakiye:", bakiye, "TL")

                else:
                    print("Bot: Transfer iptal edildi.")

        except ValueError:
            print("Bot: Geçersiz tutar girdiniz.")

    # KREDI KARTI
    elif niyet == "kredi_karti":

        print("Kredi kartı limitiniz:", kart_limiti, "TL")

        if kart_bloke:
            print("Kartınız bloke durumundadır.")

    # KAYIP KART
    elif niyet == "kayip_kart":

        if kart_bloke:
            print("Kartınız zaten bloke edilmiş.")

        else:
            print("Kart kaybı bildirimi alındı.")

            onay = input("Kart bloke edilsin mi? (e/h): ")

            if onay.lower() == "e":

                kart_bloke = True

                print("Bot: Demo kartınız bloke edildi.")

            else:
                print("Bot: İşlem iptal edildi.")

    print()
