# UYGULAMA: Gelişmiş Kural Tabanlı Chatbot

import random
from datetime import datetime

# Botun verebileceği rastgele cevaplar
selamlar = [
    "Merhaba!",
    "Selam! Nasılsın?",
    "Merhaba, sana nasıl yardımcı olabilirim?"
]

nasilsin_cevaplari = [
    "İyiyim, teşekkür ederim!",
    "Harikayım! Sen nasılsın?",
    "Gayet iyiyim, bugün neler yapıyoruz?"
]

# Chatbot başlangıcı
print("=== PYTHON CHATBOT ===")

isim = input("Adınız nedir? ").strip()

if isim == "":
    isim = "Arkadaşım"

print("Merhaba", isim)
print("Ben Python ile geliştirilmiş bir chatbotum.")
print("Neler yapabildiğimi öğrenmek için 'yardım' yaz.")
print("Çıkmak için 'q' yaz.\n")

# Sohbet döngüsü
while True:

    mesaj = input(isim + ": ").strip().lower()

    # Programdan çıkış
    if mesaj == "q":
        print("Bot: Görüşmek üzere", isim)
        break

    # Boş mesaj kontrolü
    elif mesaj == "":
        print("Bot: Lütfen bir mesaj yaz.")

    # Yardım menüsü
    elif mesaj == "yardım":
        print("\nBot: Şunları yapabilirim:")
        print("- Selamlaşabilirim")
        print("- Adını hatırlayabilirim")
        print("- Saat ve tarih söyleyebilirim")
        print("- Python hakkında bilgi verebilirim")
        print("- Yapay zekayı açıklayabilirim")
        print("- Toplama işlemi yapabilirim")
        print("- Şaka yapabilirim")
        print("- Motivasyon sözü söyleyebilirim\n")

    # Selamlaşma
    elif "merhaba" in mesaj or "selam" in mesaj:
        print("Bot:", random.choice(selamlar))

    # Nasılsın sorusu
    elif "nasılsın" in mesaj:
        print("Bot:", random.choice(nasilsin_cevaplari))

    # Kullanıcının adını hatırlama
    elif "adım ne" in mesaj or "ismim ne" in mesaj:
        print("Bot: Senin adın", isim)

    # Botun adını öğrenme
    elif "adın ne" in mesaj:
        print("Bot: Benim adım PythonBot!")

    # Saat
    elif "saat" in mesaj:
        saat = datetime.now().strftime("%H:%M")
        print("Bot: Şu an saat", saat)

    # Tarih
    elif "tarih" in mesaj or "bugün günlerden" in mesaj:
        bugun = datetime.now().strftime("%d.%m.%Y")
        print("Bot: Bugünün tarihi:", bugun)

    # Python hakkında bilgi
    elif "python" in mesaj:
        print("Bot: Python, öğrenmesi kolay ve güçlü bir programlama dilidir.")

    # Yapay zeka hakkında bilgi
    elif "yapay zeka" in mesaj or "yapay zekâ" in mesaj:
        print("Bot: Yapay zeka, bilgisayarların öğrenme ve karar verme gibi görevleri gerçekleştirmesini sağlayan teknolojilerdir.")

    # Toplama işlemi
    elif "toplama" in mesaj or "topla" in mesaj:
        try:
            sayi1 = float(input("Bot: Birinci sayıyı gir: "))
            sayi2 = float(input("Bot: İkinci sayıyı gir: "))

            toplam = sayi1 + sayi2
            print("Bot: Sonuç =", toplam)

        except ValueError:
            print("Bot: Lütfen geçerli bir sayı gir.")

    # Şaka
    elif "şaka" in mesaj:
        sakalar = [
            "Bilgisayar neden üşümüş? Windows açık kalmış!",
            "Programcı neden gözlük takar? Çünkü C# göremiyordur!",
            "Bilgisayar neden doktora gitmiş? Virüs kapmış!"
        ]

        print("Bot:", random.choice(sakalar))

    # Motivasyon sözü
    elif "motivasyon" in mesaj:
        sozler = [
            "Başarı küçük adımlarla başlar.",
            "Hata yapmak öğrenmenin bir parçasıdır.",
            "Bugün öğrendiğin bilgi yarın işine yarayacak.",
            "Pes etme, her deneme seni geliştirir."
        ]

        print("Bot:", random.choice(sozler))

    # Teşekkür
    elif "teşekkür" in mesaj or "sağ ol" in mesaj:
        print("Bot: Rica ederim", isim)

    # Anlaşılamayan mesajlar
    else:
        print("Bot: Üzgünüm, ne demek istediğini anlayamadım.")
        print("Bot: Yapabileceklerimi görmek için 'yardım' yaz.")
