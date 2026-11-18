
# GORSEL BANKACILIK CHATBOTU
# Streamlit + TF-IDF + Naive Bayes

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sayfa ayarları
st.set_page_config(
    page_title="DemoBank AI",
    page_icon="🏦"
)

# Başlık
st.title("🏦 DemoBank AI")
st.write("Yapay zeka destekli bankacılık asistanı")
st.info("Bu uygulama eğitim amaçlıdır. Gerçek banka işlemi yapmaz.")

# Eğitim verileri
sorular = [
    "Bakiyem ne kadar",
    "Hesabımda kaç para var",
    "Hesap bakiyemi göster",
    "Ne kadar param kaldı",

    "Son işlemlerimi göster",
    "Hesap hareketlerim neler",
    "Son harcamalarım",
    "Hesap geçmişimi göster",

    "Kart limitim ne kadar",
    "Kredi kartı limitimi göster",
    "Kartımın limiti nedir",
    "Kullanılabilir kart limiti",

    "Kartımı kaybettim",
    "Kartım çalındı",
    "Kartımı bloke etmek istiyorum",
    "Kayıp kart bildirimi",

    "Para göndermek istiyorum",
    "Havale nasıl yapılır",
    "Para transferi yapmak istiyorum",
    "EFT işlemi yapmak istiyorum"
]

etiketler = (
    ["bakiye"] * 4 +
    ["hareketler"] * 4 +
    ["kart_limiti"] * 4 +
    ["kayip_kart"] * 4 +
    ["transfer"] * 4
)

# Modeli oluşturma
@st.cache_resource
def modeli_egit():
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(sorular)

    model = MultinomialNB(alpha=0.5)
    model.fit(X, etiketler)

    return vec, model

vec, model = modeli_egit()

# Örnek hesap bilgileri
if "bakiye" not in st.session_state:
    st.session_state.bakiye = 12500

if "kart_bloke" not in st.session_state:
    st.session_state.kart_bloke = False

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [
        {
            "role": "assistant",
            "content": "Merhaba! Size nasıl yardımcı olabilirim?"
        }
    ]

# Yan menü
with st.sidebar:
    st.header("🏦 DemoBank")

    st.subheader("Hesap Bilgileri")
    st.metric("Bakiye", f"{st.session_state.bakiye} TL")
    st.metric("Kart Limiti", "20000 TL")

    st.write("Kart bloke:", st.session_state.kart_bloke)

    st.divider()
    st.write("Örnek sorular:")
    st.write("• Bakiyem ne kadar?")
    st.write("• Son işlemlerimi göster")
    st.write("• Kart limitim nedir?")
    st.write("• Kartımı kaybettim")
    st.write("• Para göndermek istiyorum")

    if st.button("Sohbeti Temizle"):
        st.session_state.mesajlar = []
        st.rerun()

# Önceki mesajları göster
for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["role"]):
        st.write(mesaj["content"])

# Kullanıcıdan mesaj alma
soru = st.chat_input("Mesajınızı yazın...")

if soru:

    # Kullanıcı mesajını kaydet
    st.session_state.mesajlar.append({
        "role": "user",
        "content": soru
    })

    with st.chat_message("user"):
        st.write(soru)

    # Metni sayısallaştır
    yeni_soru = vec.transform([soru])

    if yeni_soru.nnz == 0:
        cevap = "Sorunuzu anlayamadım. Lütfen bankacılık işlemi sorunuz."

    else:
        # Niyet tahmini
        niyet = model.predict(yeni_soru)[0]

        if niyet == "bakiye":
            cevap = f"Güncel bakiyeniz: {st.session_state.bakiye} TL"

        elif niyet == "hareketler":
            cevap = (
                "Son işlemleriniz:\n\n"
                "• Market: -350 TL\n\n"
                "• Fatura: -750 TL\n\n"
                "• Maaş: +25000 TL"
            )

        elif niyet == "kart_limiti":
            cevap = "Kredi kartı limitiniz 20000 TL."

        elif niyet == "kayip_kart":
            cevap = (
                "Kartınızı kaybettiğinizi belirttiniz. "
                "Bu eğitim uygulamasında gerçek kart bloke işlemi yapılamaz."
            )

        elif niyet == "transfer":
            cevap = (
                "Para transferi talebinizi anladım. "
                "Bu uygulamada gerçek para gönderimi yapılmaz."
            )

        else:
            cevap = "Size nasıl yardımcı olabilirim?"

    # Bot cevabını kaydet
    st.session_state.mesajlar.append({
        "role": "assistant",
        "content": cevap
    })

    with st.chat_message("assistant"):
        st.write(cevap)
