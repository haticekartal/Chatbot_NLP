
# UYGULAMA: CAREER AI - KARIYER DANISMANI
# Streamlit + TF-IDF + Naive Bayes

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# 1. SAYFA AYARLARI

st.set_page_config(
    page_title="CareerAI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CareerAI")
st.subheader("Yapay Zekâ Destekli Kariyer Danışmanı")

st.write(
    "Meslek seçimi, CV hazırlama, iş görüşmeleri "
    "ve kariyer planlaması hakkında sorular sorabilirsiniz."
)

st.info(
    "Bu uygulama eğitim amaçlıdır. "
    "Öneriler gerçek bir kariyer danışmanının "
    "kişisel değerlendirmesi yerine geçmez."
)

# 2. EGITIM VERILERI

veriler = {
    "meslek_onerisi": [
        "Hangi mesleği seçmeliyim",
        "Bana uygun meslek nedir",
        "Hangi işte çalışabilirim",
        "Yeteneklerime göre meslek öner",
        "Becerilerime uygun kariyer alanı",
        "Ne iş yapabilirim",
        "Hangi alanda çalışmalıyım",
        "Kendime uygun meslek arıyorum"
    ],

    "cv": [
        "CV nasıl hazırlanır",
        "Özgeçmiş nasıl yazılır",
        "CV hazırlamak istiyorum",
        "CV için öneri ver",
        "Özgeçmişime neler eklemeliyim",
        "İyi bir CV nasıl olmalı",
        "CV hazırlarken nelere dikkat edilmeli",
        "İş başvurusu için özgeçmiş"
    ],

    "mulakat": [
        "Mülakata nasıl hazırlanırım",
        "İş görüşmesinde ne sorulur",
        "Mülakat soruları nelerdir",
        "İş görüşmesi için tavsiye",
        "Mülakatta kendimi nasıl tanıtmalıyım",
        "İş görüşmesinde nasıl davranmalıyım",
        "Mülakat örnek sorular",
        "İşe alım görüşmesi hakkında bilgi"
    ],

    "beceri": [
        "Hangi becerileri öğrenmeliyim",
        "Kendimi nasıl geliştirebilirim",
        "Hangi programlama dillerini öğrenmeliyim",
        "Teknik becerilerimi geliştirmek istiyorum",
        "Hangi kurslara katılabilirim",
        "Yazılım becerilerimi nasıl artırabilirim",
        "İş hayatında hangi beceriler önemli",
        "Kariyer için hangi teknolojileri öğrenmeliyim"
    ],

    "kariyer_plani": [
        "Kariyer planı nasıl yapılır",
        "Kariyer yol haritası oluştur",
        "Yazılımcı olmak için ne yapmalıyım",
        "Veri analisti olmak için yol haritası",
        "Mesleğe nasıl hazırlanabilirim",
        "Kariyerime nereden başlamalıyım",
        "İş hayatına nasıl hazırlanılır",
        "Kariyer hedefimi nasıl belirlemeliyim"
    ],

    "is_arama": [
        "Nasıl iş bulabilirim",
        "İş ararken nelere dikkat etmeliyim",
        "İş başvurusu nasıl yapılır",
        "İş ilanlarına nasıl başvurulur",
        "Staj nasıl bulabilirim",
        "İş arama yöntemleri nelerdir",
        "Yeni mezun olarak nasıl iş bulurum",
        "İş başvurularında nasıl başarılı olurum"
    ]
}

# 3. SORULARI VE ETIKETLERI OLUSTURMA

sorular = []
etiketler = []

for konu, ornek_sorular in veriler.items():

    for soru in ornek_sorular:
        sorular.append(soru)
        etiketler.append(konu)

# 4. MODELI EGITME

@st.cache_resource
def modeli_egit():

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        stop_words=[
            "bir", "ve", "ile", "için",
            "nasıl", "nedir", "ne", "hangi",
            "ben", "bana", "istiyorum"
        ]
    )

    X = vec.fit_transform(sorular)

    model = MultinomialNB(alpha=0.5)

    model.fit(X, etiketler)

    return vec, model, X


vec, model, X_egitim = modeli_egit()

# 5. MESLEK BILGILERI

meslekler = {
    "Veri Analisti": {
        "kelimeler": [
            "python", "sql", "excel",
            "pandas", "veri", "analiz", "power bi"
        ],
        "aciklama": (
            "Verileri analiz eder, raporlar "
            "ve grafiklerle sonuçları açıklar."
        )
    },

    "Yazılım Geliştirici": {
        "kelimeler": [
            "python", "java", "c#",
            "yazılım", "kodlama", "algoritma"
        ],
        "aciklama": (
            "Web, masaüstü veya mobil "
            "uygulamalar geliştirir."
        )
    },

    "İş Analisti": {
        "kelimeler": [
            "analiz", "iletişim", "sql",
            "iş süreçleri", "gereksinim", "excel"
        ],
        "aciklama": (
            "İş ihtiyaçlarını belirler ve "
            "teknoloji ekipleriyle çözüm geliştirir."
        )
    },

    "Yapay Zekâ Uzmanı": {
        "kelimeler": [
            "python", "makine öğrenmesi",
            "yapay zeka", "tensorflow",
            "veri", "matematik"
        ],
        "aciklama": (
            "Makine öğrenmesi ve yapay zekâ "
            "modelleri geliştirir."
        )
    },

    "Grafik Tasarımcı": {
        "kelimeler": [
            "tasarım", "canva", "photoshop",
            "görsel", "renk", "figma"
        ],
        "aciklama": (
            "Dijital tasarımlar ve "
            "görsel içerikler oluşturur."
        )
    },

    "Siber Güvenlik Uzmanı": {
        "kelimeler": [
            "güvenlik", "ağ", "linux",
            "siber", "network", "python"
        ],
        "aciklama": (
            "Bilgi sistemlerinin güvenliğini "
            "sağlamak için çalışır."
        )
    }
}

# 6. MESLEK ONERI FONKSIYONU

def meslek_oner(beceriler):

    beceriler = beceriler.lower()

    sonuclar = []

    for meslek, bilgiler in meslekler.items():

        puan = 0

        for kelime in bilgiler["kelimeler"]:

            if kelime in beceriler:
                puan += 1

        if puan > 0:
            sonuclar.append(
                (meslek, puan, bilgiler["aciklama"])
            )

    # Puanı yüksek meslekleri öne al
    sonuclar.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if len(sonuclar) == 0:
        return (
            "Becerilerinize göre öneri bulamadım. "
            "Python, SQL, Excel, tasarım, "
            "iletişim gibi becerilerinizi yazabilirsiniz."
        )

    cevap = "🎯 Becerilerinize göre örnek meslekler:\n\n"

    for meslek, puan, aciklama in sonuclar[:3]:

        cevap += f"**{meslek}**\n\n"
        cevap += f"{aciklama}\n\n"
        cevap += f"Eşleşen beceri sayısı: {puan}\n\n"

    cevap += (
        "Bu sonuçlar yalnızca anahtar kelime "
        "eşleşmesine dayanır; kesin mesleki "
        "uygunluk değerlendirmesi değildir."
    )

    return cevap

# 7. HAZIR CEVAPLAR

cevaplar = {
    "cv": (
        "📄 **CV Hazırlama Önerileri**\n\n"
        "1. Adınızı ve iletişim bilgilerinizi yazın.\n\n"
        "2. Kısa bir kariyer özeti ekleyin.\n\n"
        "3. Eğitim bilgilerinizi belirtin.\n\n"
        "4. Staj ve iş deneyimlerinizi yazın.\n\n"
        "5. Teknik becerilerinizi listeleyin.\n\n"
        "6. Projelerinizi ve sertifikalarınızı ekleyin.\n\n"
        "7. CV'nizi başvurduğunuz pozisyona göre düzenleyin."
    ),

    "mulakat": (
        "💼 **Mülakat Hazırlığı**\n\n"
        "İş görüşmelerinde şu sorularla "
        "karşılaşabilirsiniz:\n\n"
        "1. Kendinizden bahseder misiniz?\n\n"
        "2. Neden bu pozisyonda çalışmak istiyorsunuz?\n\n"
        "3. Güçlü yönleriniz nelerdir?\n\n"
        "4. Zor bir problemi nasıl çözdünüz?\n\n"
        "5. Takım çalışması deneyiminiz var mı?\n\n"
        "6. Beş yıl sonra kendinizi nerede görüyorsunuz?\n\n"
        "Öneri: Cevaplarınızı gerçek deneyimlerinizle "
        "destekleyin ve STAR yöntemini kullanın."
    ),

    "beceri": (
        "🧠 **Kendinizi Geliştirebileceğiniz Alanlar**\n\n"
        "• Python programlama\n\n"
        "• SQL ve veritabanı\n\n"
        "• Excel ve veri analizi\n\n"
        "• Yapay zekâ ve makine öğrenmesi\n\n"
        "• İngilizce\n\n"
        "• Problem çözme ve iletişim\n\n"
        "İlgilendiğiniz mesleğe göre "
        "öğrenme önceliklerinizi belirleyin."
    ),

    "kariyer_plani": (
        "🗺️ **Örnek Kariyer Yol Haritası**\n\n"
        "1. İlgi alanlarınızı belirleyin.\n\n"
        "2. Hedeflediğiniz mesleği araştırın.\n\n"
        "3. Gerekli teknik becerileri öğrenin.\n\n"
        "4. Küçük projeler geliştirin.\n\n"
        "5. GitHub veya portföy oluşturun.\n\n"
        "6. CV hazırlayın.\n\n"
        "7. Staj ve iş başvuruları yapın.\n\n"
        "8. Mülakat pratiği yapın."
    ),

    "is_arama": (
        "🔎 **İş Arama Önerileri**\n\n"
        "1. Güncel bir CV hazırlayın.\n\n"
        "2. LinkedIn profilinizi düzenleyin.\n\n"
        "3. İlgilendiğiniz iş ilanlarını inceleyin.\n\n"
        "4. Her başvuru için CV'nizi uyarlayın.\n\n"
        "5. Projelerinizi GitHub'da paylaşın.\n\n"
        "6. Mülakat sorularına çalışın.\n\n"
        "7. Başvurularınızı bir tabloda takip edin."
    )
}

# 8. SOHBET CEVAP FONKSIYONU

def cevap_uret(soru, beceriler):

    yeni_soru = vec.transform([soru])

    # Soru bilinen hiçbir kelimeyi içermiyorsa
    if yeni_soru.nnz == 0:
        return (
            "Bu soruyu anlayamadım. "
            "CV, mülakat, meslek seçimi veya "
            "kariyer planlaması hakkında sorabilirsiniz."
        )

    # Eğitim sorularıyla benzerlik
    benzerlikler = cosine_similarity(
        yeni_soru, X_egitim
    )[0]

    en_benzer = max(benzerlikler)

    # Çok düşük benzerlikte cevap vermeme
    if en_benzer < 0.15:
        return (
            "Sorunuzu tam olarak anlayamadım. "
            "Daha açık bir şekilde sorabilir misiniz?"
        )

    # Naive Bayes ile niyet tahmini
    niyet = model.predict(yeni_soru)[0]

    if niyet == "meslek_onerisi":

        if beceriler.strip() == "":
            return (
                "Size meslek önerebilmem için "
                "sol menüdeki beceriler alanını doldurun.\n\n"
                "Örnek: Python, SQL, Excel, iletişim"
            )

        return meslek_oner(beceriler)

    return cevaplar[niyet]

# 9. OTURUM BILGILERI

if "mesajlar" not in st.session_state:

    st.session_state.mesajlar = [
        {
            "role": "assistant",
            "content": (
                "Merhaba! 👋 Ben CareerAI.\n\n"
                "Meslek seçimi, CV hazırlama ve "
                "iş görüşmeleri konusunda "
                "size yardımcı olabilirim."
            )
        }
    ]

# 10. SOL MENU

with st.sidebar:

    st.header("👤 Kariyer Profilim")

    isim = st.text_input(
        "Adınız",
        placeholder="Örnek: Ayşe"
    )

    beceriler = st.text_area(
        "Becerileriniz",
        placeholder="Python, SQL, Excel, iletişim..."
    )

    egitim = st.selectbox(
        "Eğitim Durumunuz",
        [
            "Seçiniz",
            "Lise",
            "Ön Lisans",
            "Lisans",
            "Yüksek Lisans"
        ]
    )

    st.divider()

    st.subheader("💡 Örnek Sorular")

    st.write("• Hangi mesleği seçmeliyim?")
    st.write("• CV nasıl hazırlanır?")
    st.write("• Mülakatta ne sorulur?")
    st.write("• Nasıl iş bulabilirim?")
    st.write("• Kariyer planı nasıl yapılır?")

    st.divider()

    if st.button("🗑️ Sohbeti Temizle"):

        st.session_state.mesajlar = []

        st.rerun()

# 11. KULLANICI BILGILERI

if isim:
    st.success(f"Hoş geldiniz, {isim}!")

# 12. SOHBET GECMISI

for mesaj in st.session_state.mesajlar:

    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])

# 13. YENI MESAJ ALMA

soru = st.chat_input("Kariyer hakkında bir soru sorun...")

if soru:

    # Kullanıcı mesajını kaydet
    st.session_state.mesajlar.append({
        "role": "user",
        "content": soru
    })

    with st.chat_message("user"):
        st.write(soru)

    # Modelden cevap al
    cevap = cevap_uret(soru, beceriler)

    # Bot cevabını kaydet
    st.session_state.mesajlar.append({
        "role": "assistant",
        "content": cevap
    })

    with st.chat_message("assistant"):
        st.markdown(cevap)
