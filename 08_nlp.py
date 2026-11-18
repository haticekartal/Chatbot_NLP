
print("Chatbot'a hoş geldiniz!")
print("Çıkmak için q yazınız.")

while True:
    mesaj = input("Siz: ").lower()

    if mesaj == "q":
        print("Bot: Görüşmek üzere!")
        break

    elif "merhaba" in mesaj:
        print("Bot: Merhaba! Nasılsınız?")

    elif "nasılsın" in mesaj:
        print("Bot: İyiyim, teşekkür ederim.")

    elif "adın ne" in mesaj:
        print("Bot: Ben Python ile oluşturuldum.")

    elif "python" in mesaj:
        print("Bot: Python bir programlama dilidir.")

    elif "yapay zeka" in mesaj:
        print("Bot: Yapay zeka, insan zekası gerektiren görevleri gerçekleştiren sistemlerdir.")

    else:
        print("Bot: Bu soruya cevap veremiyorum.")
