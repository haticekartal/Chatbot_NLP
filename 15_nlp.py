
# UYGULAMA: BOOKAI - KUTUPHANE CHATBOTU
# Streamlit + TF-IDF + Naive Bayes
# Kitap onerme, arama, odunc alma ve iade

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# 1. SAYFA AYARLARI

st.set_page_config(
    page_title="BookAI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 BookAI - Akıllı Kütüphane")
st.write("Yapay zekâ destekli kitap öneri ve kütüphane asistanı")
st.caption("Eğitim amaçlı uygulama - Kitap işlemleri simülasyondur.")

# 2. KITAP VERILERI

kitaplar = [
    {
        "ad": "1984",
        "yazar": "George Orwell",
        "tur": "Bilim Kurgu",
        "konu": "Distopya, gözetim, totaliter devlet ve özgürlük"
    },
    {
        "ad": "Dune",
        "yazar": "Frank Herbert",
        "tur": "Bilim Kurgu",
        "konu": "Uzay, gezegenler, siyaset ve bilim kurgu"
    },
    {
        "ad": "Fahrenheit 451",
        "yazar": "Ray Bradbury",
        "tur": "Bilim Kurgu",
        "konu": "Distopya, sansür, kitaplar ve toplum"
    },
    {
        "ad": "Kürk Mantolu Madonna",
        "yazar": "Sabahattin Ali",
        "tur": "Roman",
        "konu": "Aşk, yalnızlık ve insan ilişkileri"
    },
    {
        "ad": "Suç ve Ceza",
        "yazar": "Fyodor Dostoyevski",
        "tur": "Roman",
        "konu": "Suç, vicdan, psikoloji ve ahlak"
    },
    {
        "ad": "Simyacı",
        "yazar": "Paulo Coelho",
        "tur": "Roman",
        "konu": "Hayaller, yolculuk ve kendini keşfetme"
    },
    {
        "ad": "Sherlock Holmes",
        "yazar": "Arthur Conan Doyle",
        "tur": "Polisiye",
        "konu": "Dedektif, gizem, suç ve araştırma"
    },
    {
        "ad": "Doğu Ekspresinde Cinayet",
        "yazar": "Agatha Christie",
        "tur": "Polisiye",
        "konu": "Cinayet, dedektif, gizem ve soruşturma"
    },
    {
        "ad": "On Küçük Zenci",
        "yazar": "Agatha Christie",
        "tur": "Polisiye",
        "konu": "Gizem, cinayet, suç ve gerilim"
    },
    {
        "ad": "Sapiens",
        "yazar": "Yuval Noah Harari",
        "tur": "Tarih",
        "konu": "İnsanlık tarihi, toplum ve uygarlık"
    },
    {
        "ad": "Nutuk",
        "yazar": "Mustafa Kemal Atatürk",
        "tur": "Tarih",
        "konu": "Kurtuluş Savaşı ve Türkiye Cumhuriyeti tarihi"
    },
    {
        "ad": "İnsan Ne ile Yaşar",
        "yazar": "Lev Tolstoy",
        "tur": "Felsefe",
        "konu": "İnsanlık, iyilik, sevgi ve hayatın anlamı"
    },
    {
        "ad": "Sofinin Dünyası",
        "yazar": "Jostein Gaarder",
        "tur": "Felsefe",
        "konu": "Felsefe tarihi, düşünce ve bilgi"
    },
    {
        "ad": "Atomik Alışkanlıklar",
        "yazar": "James Clear",
        "tur": "Kişisel Gelişim",
        "konu": "Alışkanlık, disiplin ve hedef belirleme"
    },
    {
        "ad": "Akış",
        "yazar": "Mihaly Csikszentmihalyi",
        "tur": "Kişisel Gelişim",
        "konu": "Odaklanma, mutluluk ve motivasyon"
    }
]

# 3. CHATBOT EGITIM VERILERI

egitim = {
    "oneri": [
        "Bana kitap öner",
        "Hangi kitabı okumalıyım",
        "Bilim kurgu kitabı öner",
        "Okuyabileceğim roman öner",
        "İlgi alanıma uygun kitap bul",
        "Bana yeni kitap tavsiye et"
    ],
    "arama": [
        "Kitap aramak istiyorum",
        "1984 kitabı var mı",
        "Kütüphanede kitap ara",
        "Bir kitabın varlığını kontrol et",
        "Kitap bulmak istiyorum",
        "Sabahattin Ali kitaplarını göster"
    ],
    "odunc": [
        "Kitap ödünç almak istiyorum",
        "Bir kitap alabilir miyim",
        "Kitap ödünç alma işlemi",
        "Kütüphaneden kitap almak istiyorum",
        "Kitabı ödünç verebilir misiniz",
        "Kitap almak istiyorum"
    ],
    "iade": [
        "Kitap iade etmek istiyorum",
        "Aldığım kitabı geri vereceğim",
        "Kitap iadesi nasıl yapılır",
        "Ödünç aldığım kitabı iade et",
        "Kitabı geri getirdim",
        "Kitap geri verme işlemi"
    ],
    "turler": [
        "Hangi kitap türleri var",
        "Kategorileri göster",
        "Kütüphanede hangi tür kitaplar bulunuyor",
        "Kitap kategorileri nelerdir",
        "Roman ve polisiye türleri var mı",
        "Mevcut kitap türlerini listele"
    ],
    "bilgi": [
        "Kütüphane hakkında bilgi ver",
        "Kütüphanede kaç kitap var",
        "Toplam kitap sayısı nedir",
        "Kütüphane bilgilerini göster",
        "Kaç farklı kitap bulunuyor",
        "Kütüphaneyi tanıt"
    ]
}

sorular = []
etiketler = []

for niyet, ornekler in egitim.items():
    for soru in ornekler:
        sorular.append(soru)
        etiketler.append(niyet)

# 4. NAIVE BAYES MODELINI EGITME

@st.cache_resource
def modeli_egit():

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(sorular)

    model = MultinomialNB(alpha=0.5)
    model.fit(X, etiketler)

    return vec, model, X

vec, model, X_egitim = modeli_egit()

# 5. KITAP ONERI MODELI

@st.cache_resource
def oneri_modeli():

    aciklamalar = []

    for kitap in kitaplar:
        metin = kitap["tur"] + " " + kitap["konu"]
        aciklamalar.append(metin)

    kitap_vec = TfidfVectorizer()
    kitap_X = kitap_vec.fit_transform(aciklamalar)

    return kitap_vec, kitap_X

kitap_vec, kitap_X = oneri_modeli()

# 6. OTURUM BILGILERI

if "odunc_kitaplar" not in st.session_state:
    st.session_state.odunc_kitaplar = []

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [
        {
            "role": "assistant",
            "content": (
                "Merhaba! 👋 Ben BookAI.\n\n"
                "Size kitap önerebilir, kitap arayabilir "
                "ve kütüphane işlemlerinizde yardımcı olabilirim."
            )
        }
    ]

# 7. KITAP ARAMA FONKSIYONU

def kitap_ara(metin):

    sonuclar = []

    for kitap in kitaplar:

        if (
            kitap["ad"].casefold() in metin.casefold()
            or kitap["yazar"].casefold() in metin.casefold()
        ):
            sonuclar.append(kitap)

    return sonuclar

# 8. KITAP ONERI FONKSIYONU

def kitap_oner(ilgi_alani):

    if not ilgi_alani.strip():
        return "Lütfen ilgilendiğiniz kitap türünü belirtin."

    # Kullanıcının ilgi alanını sayısallaştır
    yeni_veri = kitap_vec.transform([ilgi_alani])

    if yeni_veri.nnz == 0:
        return (
            "Bu ilgi alanıyla eşleşen bir konu bulamadım. "
            "Bilim kurgu, aşk, tarih, gizem veya "
            "felsefe gibi konular deneyebilirsiniz."
        )

    # Kosinüs benzerliği
    benzerlikler = cosine_similarity(
        yeni_veri, kitap_X
    )[0]

    # Benzerlik puanına göre sıralama
    siralama = benzerlikler.argsort()[::-1]

    cevap = "📚 **Size önerdiğim kitaplar:**\n\n"
    sayac = 0

    for indeks in siralama:

        if benzerlikler[indeks] <= 0:
            continue

        kitap = kitaplar[indeks]

        cevap += f"**{kitap['ad']}**\n\n"
        cevap += f"✍️ Yazar: {kitap['yazar']}\n\n"
        cevap += f"📖 Tür: {kitap['tur']}\n\n"
        cevap += f"📝 Konu: {kitap['konu']}\n\n"
        cevap += "---\n\n"

        sayac += 1

        if sayac == 3:
            break

    return cevap

# 9. CHATBOT CEVAP FONKSIYONU

def cevap_uret(soru, ilgi_alani):

    yeni_soru = vec.transform([soru])

    if yeni_soru.nnz == 0:
        return (
            "Sorunuzu anlayamadım. "
            "Kitap önerisi, arama veya ödünç alma "
            "hakkında soru sorabilirsiniz."
        )

    # Eğitim sorularına benzerliği ölç
    benzerlikler = cosine_similarity(
        yeni_soru, X_egitim
    )[0]

    if max(benzerlikler) < 0.15:
        return "Sorunuzu biraz daha açık yazabilir misiniz?"

    # Kullanıcının niyetini tahmin et
    niyet = model.predict(yeni_soru)[0]

    if niyet == "oneri":

        # Kullanıcının sorusundaki türü öncelikle kullan
        turler = sorted(
            set(k["tur"] for k in kitaplar)
        )

        secilen_ilgi = ilgi_alani

        for tur in turler:
            if tur.casefold() in soru.casefold():
                secilen_ilgi = tur
                break

        if not secilen_ilgi:
            return (
                "Size öneride bulunabilmem için "
                "sol menüden ilgi alanınızı seçin."
            )

        return kitap_oner(secilen_ilgi)

    elif niyet == "arama":

        bulunanlar = kitap_ara(soru)

        if len(bulunanlar) == 0:
            return (
                "Hangi kitabı veya yazarı arıyorsunuz?\n\n"
                "Kitap adını yazarak tekrar sorabilirsiniz."
            )

        cevap = "🔎 **Arama sonuçları:**\n\n"

        for kitap in bulunanlar:

            ad = kitap["ad"]

            if ad in st.session_state.odunc_kitaplar:
                durum = "Ödünç alınmış"
            else:
                durum = "Mevcut"

            cevap += (
                f"**{ad}** - {kitap['yazar']}\n\n"
                f"Durum: {durum}\n\n"
            )

        return cevap

    elif niyet == "odunc":
        return (
            "📖 Kitap ödünç almak için sol menüdeki "
            "'Kitap Ödünç Alma' bölümünü kullanabilirsiniz."
        )

    elif niyet == "iade":
        return (
            "📚 Kitap iade etmek için sol menüdeki "
            "'Kitap İade Etme' bölümünü kullanabilirsiniz."
        )

    elif niyet == "turler":

        turler = sorted(
            set(kitap["tur"] for kitap in kitaplar)
        )

        return "📚 Kitap türleri:\n\n" + "\n\n".join(
            "- " + tur for tur in turler
        )

    elif niyet == "bilgi":

        return (
            f"Kütüphanemizde {len(kitaplar)} farklı "
            "kitap bulunmaktadır.\n\n"
            f"Ödünç alınan kitap sayısı: "
            f"{len(st.session_state.odunc_kitaplar)}"
        )

    return "Size nasıl yardımcı olabilirim?"

# 10. SOL MENU

with st.sidebar:

    st.header("📚 BookAI Kütüphanesi")

    isim = st.text_input(
        "Adınız",
        placeholder="Adınızı yazınız"
    )

    turler = sorted(
        set(kitap["tur"] for kitap in kitaplar)
    )

    ilgi_alani = st.selectbox(
        "İlgi alanınız",
        ["Seçiniz"] + turler
    )

    if ilgi_alani == "Seçiniz":
        ilgi_alani = ""

    if st.button("🎯 Bana Kitap Öner"):

        cevap = kitap_oner(ilgi_alani)

        st.session_state.mesajlar.append({
            "role": "assistant",
            "content": cevap
        })

        st.rerun()

    st.divider()

    st.subheader("📖 Kitap Ödünç Alma")

    kitap_adlari = [k["ad"] for k in kitaplar]

    secilen_kitap = st.selectbox(
        "Kitap seçiniz",
        kitap_adlari
    )

    if st.button("Kitabı Ödünç Al"):

        if secilen_kitap in st.session_state.odunc_kitaplar:

            st.warning("Bu kitap zaten ödünç alınmış.")

        else:

            st.session_state.odunc_kitaplar.append(
                secilen_kitap
            )

            st.success(
                f"{secilen_kitap} ödünç alındı."
            )

    st.divider()

    st.subheader("↩️ Kitap İade Etme")

    if st.session_state.odunc_kitaplar:

        iade_kitap = st.selectbox(
            "İade edilecek kitap",
            st.session_state.odunc_kitaplar
        )

        if st.button("Kitabı İade Et"):

            st.session_state.odunc_kitaplar.remove(
                iade_kitap
            )

            st.success("Kitap iade edildi.")
            st.rerun()

    else:
        st.write("Ödünç alınmış kitabınız yok.")

    st.divider()

    if st.button("🗑️ Sohbeti Temizle"):

        st.session_state.mesajlar = []
        st.rerun()

# 11. KUTUPHANE BILGILERI

sutun1, sutun2, sutun3 = st.columns(3)

with sutun1:
    st.metric("Toplam Kitap", len(kitaplar))

with sutun2:
    st.metric(
        "Mevcut Kitap",
        len(kitaplar) - len(st.session_state.odunc_kitaplar)
    )

with sutun3:
    st.metric(
        "Ödünç Alınan",
        len(st.session_state.odunc_kitaplar)
    )

st.divider()

# 12. SOHBET EKRANI

if isim:
    st.success(f"Hoş geldiniz, {isim}!")

for mesaj in st.session_state.mesajlar:

    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])

# 13. KULLANICIDAN MESAJ ALMA

soru = st.chat_input("Kitaplarla ilgili soru sorun...")

if soru:

    st.session_state.mesajlar.append({
        "role": "user",
        "content": soru
    })

    with st.chat_message("user"):
        st.write(soru)

    cevap = cevap_uret(soru, ilgi_alani)

    st.session_state.mesajlar.append({
        "role": "assistant",
        "content": cevap
    })

    with st.chat_message("assistant"):
        st.markdown(cevap)
