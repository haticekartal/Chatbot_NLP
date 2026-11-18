
# UYGULAMA: SENTIMENT AI - DUYGU ANALIZI
# Streamlit + TF-IDF + Naive Bayes

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. SAYFA AYARLARI

st.set_page_config(
    page_title="SentimentAI",
    page_icon="😊",
    layout="wide"
)

st.title("😊 SentimentAI")
st.subheader("Yapay Zekâ Destekli Duygu Analizi")

st.write(
    "Müşteri yorumlarını yapay zekâ ile analiz edin. "
    "Yorumların olumlu, olumsuz veya nötr olduğunu öğrenin."
)

st.info(
    "Bu uygulama eğitim amaçlıdır. "
    "Model sınırlı sayıda örnek yorumla eğitilmektedir."
)

# 2. EGITIM VERILERI

yorumlar = [
    # OLUMLU YORUMLAR
    "Ürün harika çok beğendim",
    "Kargo çok hızlı geldi",
    "Ürünün kalitesi mükemmel",
    "Çok memnun kaldım",
    "Kesinlikle tavsiye ederim",
    "Paketleme çok güzeldi",
    "Fiyatına göre çok başarılı",
    "Beklediğimden daha iyi çıktı",
    "Satıcı çok ilgiliydi",
    "Ürün tam istediğim gibi",
    "Hizmet gerçekten çok iyiydi",
    "Alışverişten çok memnunum",
    "Kalitesi beklentilerimi aştı",
    "Teslimat çok hızlıydı",
    "Her şey çok güzeldi",

    # OLUMSUZ YORUMLAR
    "Ürün çok kötü hiç beğenmedim",
    "Kargo çok geç geldi",
    "Ürünün kalitesi berbat",
    "Hiç memnun kalmadım",
    "Kesinlikle tavsiye etmiyorum",
    "Paketleme çok kötüydü",
    "Verdiğim paraya değmedi",
    "Beklediğimden daha kötü çıktı",
    "Satıcı hiç ilgilenmedi",
    "Ürün bozuk geldi",
    "Hizmet gerçekten çok kötüydü",
    "Alışverişten hiç memnun değilim",
    "Kalitesi beklentilerimin altında kaldı",
    "Teslimat çok gecikti",
    "Her şey çok kötüydü",

    # NOTR YORUMLAR
    "Ürün bugün teslim edildi",
    "Kargo sabah saatlerinde geldi",
    "Paketin içinde iki ürün vardı",
    "Ürünün rengi mavidir",
    "Sipariş dün oluşturuldu",
    "Ürün standart boyutlardadır",
    "Paket teslimat noktasına bırakıldı",
    "Ürün internet üzerinden satın alındı",
    "Sipariş numarası sisteme kaydedildi",
    "Ürün kutunun içinde bulunuyor",
    "Teslimat tarihi pazartesi olarak görünüyor",
    "Ürünün ağırlığı iki kilogram",
    "Sipariş bilgileri güncellendi",
    "Paket üzerinde ürün kodu yazıyor",
    "Ürün farklı renk seçeneklerine sahip"
]

# 3. ETIKETLER

etiketler = (
    ["Olumlu"] * 15 +
    ["Olumsuz"] * 15 +
    ["Nötr"] * 15
)

# 4. MODELI EGITME

@st.cache_resource
def modeli_egit():

    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        yorumlar,
        etiketler,
        test_size=0.25,
        random_state=42,
        stratify=etiketler
    )

    # TF-IDF
    vec_test = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True
    )

    X_train_v = vec_test.fit_transform(X_train)
    X_test_v = vec_test.transform(X_test)

    # Naive Bayes
    test_model = MultinomialNB(alpha=0.5)
    test_model.fit(X_train_v, y_train)

    # Test başarısı
    tahminler = test_model.predict(X_test_v)

    basari = accuracy_score(y_test, tahminler)

    rapor = classification_report(
        y_test,
        tahminler,
        zero_division=0
    )

    # Uygulamada kullanmak için tüm verilerle eğit
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True
    )

    X = vec.fit_transform(yorumlar)

    model = MultinomialNB(alpha=0.5)
    model.fit(X, etiketler)

    return vec, model, basari, rapor


vec, model, basari, rapor = modeli_egit()

# 5. DUYGU ANALIZI FONKSIYONU

def duygu_analiz_et(metin):

    # Metni sayısal veriye dönüştür
    yeni_veri = vec.transform([metin])

    # Tanınan kelime yoksa
    if yeni_veri.nnz == 0:
        return "Bilinmiyor", 0

    # Tahmin yap
    tahmin = model.predict(yeni_veri)[0]

    # Olasılıkları hesapla
    olasiliklar = model.predict_proba(yeni_veri)[0]

    en_yuksek = max(olasiliklar)

    return tahmin, en_yuksek

# 6. OTURUM BILGILERI

if "sonuclar" not in st.session_state:
    st.session_state.sonuclar = []

# 7. SOL MENU

with st.sidebar:

    st.header("🧠 Model Bilgileri")

    st.write("Algoritma: Naive Bayes")
    st.write("Özellik çıkarma: TF-IDF")
    st.write("N-gram: 1 ve 2")

    st.metric(
        "Test Başarı Oranı",
        f"%{basari * 100:.1f}"
    )

    st.caption(
        "Bu oran yalnızca küçük bir test kümesinde "
        "ölçülmüştür. Gerçek dünya performansı değildir."
    )

    st.divider()

    st.subheader("📖 Duygu Sınıfları")

    st.write("😊 Olumlu")
    st.write("😞 Olumsuz")
    st.write("😐 Nötr")

    st.divider()

    if st.button("🗑️ Analizleri Temizle"):
        st.session_state.sonuclar = []
        st.rerun()

    with st.expander("Model Değerlendirme Raporu"):
        st.text(rapor)

# 8. TEK YORUM ANALIZI

st.header("💬 Yorum Analizi")

ornekler = [
    "Kendi yorumumu yazacağım",
    "Ürün harika çok beğendim",
    "Kargo çok geç geldi",
    "Ürün bugün teslim edildi",
    "Bu alışverişten hiç memnun kalmadım"
]

secim = st.selectbox(
    "Örnek bir yorum seçin:",
    ornekler
)

if secim == ornekler[0]:
    varsayilan = ""
else:
    varsayilan = secim

yorum = st.text_area(
    "Analiz etmek istediğiniz yorumu yazın:",
    value=varsayilan,
    height=120,
    placeholder="Örneğin: Ürün çok güzel, çok memnun kaldım."
)

if st.button("🔍 Yorumu Analiz Et", type="primary"):

    if yorum.strip() == "":
        st.warning("Lütfen bir yorum yazınız.")

    else:

        duygu, olasilik = duygu_analiz_et(yorum)

        if duygu == "Olumlu":
            st.success("😊 OLUMLU YORUM")

        elif duygu == "Olumsuz":
            st.error("😞 OLUMSUZ YORUM")

        elif duygu == "Nötr":
            st.info("😐 NÖTR YORUM")

        else:
            st.warning("Yorum analiz edilemedi.")

        if duygu != "Bilinmiyor":

            st.write(
                f"Model tahmin olasılığı: %{olasilik * 100:.1f}"
            )

            st.progress(float(olasilik))

            # Sonucu kaydet
            st.session_state.sonuclar.append({
                "Yorum": yorum,
                "Duygu": duygu,
                "Tahmin Olasılığı": round(
                    olasilik * 100, 1
                )
            })

# 9. TOPLU YORUM ANALIZI

st.divider()

st.header("📝 Toplu Yorum Analizi")

st.write(
    "Birden fazla yorumu analiz etmek için "
    "her satıra bir yorum yazın."
)

toplu_yorumlar = st.text_area(
    "Yorumlar:",
    height=150,
    placeholder=(
        "Ürün çok güzel\n"
        "Kargo çok geç geldi\n"
        "Sipariş bugün teslim edildi"
    )
)

if st.button("📊 Tüm Yorumları Analiz Et"):

    satirlar = toplu_yorumlar.splitlines()

    sayac = 0

    for satir in satirlar:

        if satir.strip() == "":
            continue

        duygu, olasilik = duygu_analiz_et(satir)

        st.session_state.sonuclar.append({
            "Yorum": satir,
            "Duygu": duygu,
            "Tahmin Olasılığı": round(
                olasilik * 100, 1
            )
        })

        sayac += 1

    if sayac > 0:
        st.success(f"{sayac} yorum analiz edildi.")
    else:
        st.warning("Lütfen en az bir yorum giriniz.")

# 10. ANALIZ SONUCLARI

st.divider()

st.header("📋 Analiz Sonuçları")

if len(st.session_state.sonuclar) > 0:

    df = pd.DataFrame(st.session_state.sonuclar)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Duygu sayılarını hesapla
    olumlu = (df["Duygu"] == "Olumlu").sum()
    olumsuz = (df["Duygu"] == "Olumsuz").sum()
    notr = (df["Duygu"] == "Nötr").sum()

    # Sayısal göstergeler
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Toplam Yorum", len(df))
    col2.metric("😊 Olumlu", int(olumlu))
    col3.metric("😞 Olumsuz", int(olumsuz))
    col4.metric("😐 Nötr", int(notr))

    # 11. GRAFIK OLUSTURMA

    st.subheader("📊 Duygu Dağılımı")

    duygular = ["Olumlu", "Olumsuz", "Nötr"]
    sayilar = [olumlu, olumsuz, notr]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(duygular, sayilar)

    ax.set_title("Yorumların Duygu Dağılımı")
    ax.set_xlabel("Duygu")
    ax.set_ylabel("Yorum Sayısı")

    st.pyplot(fig)
    plt.close(fig)

    # 12. CSV DOSYASI INDIRME

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Sonuçları CSV Olarak İndir",
        data=csv,
        file_name="duygu_analizi_sonuclari.csv",
        mime="text/csv"
    )

else:
    st.info("Henüz analiz edilmiş yorum bulunmuyor.")

st.caption(
    "Not: Tahmin olasılığı, modelin hesapladığı "
    "sınıf olasılığıdır. Gerçek doğruluk garantisi değildir."
)
