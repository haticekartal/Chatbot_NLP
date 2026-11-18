
# UYGULAMA: YAPAY ZEKA DERS ASISTANI
# TF-IDF + Naive Bayes ile chatbot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. EĞİTİM VERİLERİ

sorular = [
    # Python
    "Python nedir",
    "Python ne işe yarar",
    "Python nerelerde kullanılır",
    "Python programlama dili nedir",
    "Python ile neler yapılabilir",

    # Değişken
    "Değişken nedir",
    "Python değişken nasıl tanımlanır",
    "Değişkenler ne işe yarar",
    "Veri saklamak için ne kullanılır",
    "Değişken tanımlama nasıl yapılır",

    # Liste
    "Liste nedir",
    "Python list nedir",
    "Listeye eleman nasıl eklenir",
    "Listeler ne işe yarar",
    "Append metodu nedir",

    # Döngü
    "Döngü nedir",
    "For döngüsü nedir",
    "While döngüsü ne işe yarar",
    "Tekrarlayan işlemler nasıl yapılır",
    "For ve while farkı nedir",

    # Fonksiyon
    "Fonksiyon nedir",
    "Python fonksiyon nasıl yazılır",
    "Def ne işe yarar",
    "Return ne işe yarar",
    "Fonksiyon nasıl tanımlanır",

    # Yapay zeka
    "Yapay zeka nedir",
    "Yapay zeka ne işe yarar",
    "Yapay zeka nerelerde kullanılır",
    "AI nedir",
    "Yapay zeka nasıl çalışır",

    # Makine öğrenmesi
    "Makine öğrenmesi nedir",
    "Machine learning nedir",
    "Makine öğrenmesi nasıl çalışır",
    "Makine öğrenmesi ne işe yarar",
    "Makine öğrenmesi algoritmaları nelerdir",

    # Derin öğrenme
    "Derin öğrenme nedir",
    "Deep learning nedir",
    "Derin öğrenme nasıl çalışır",
    "Derin öğrenme ile makine öğrenmesi farkı",
    "Derin öğrenme nerelerde kullanılır",

    # KNN
    "KNN nedir",
    "KNN algoritması nasıl çalışır",
    "En yakın komşu algoritması nedir",
    "KNN k değeri nedir",
    "KNN ile sınıflandırma nasıl yapılır",

    # Karar ağacı
    "Karar ağacı nedir",
    "Decision tree nedir",
    "Karar ağacı nasıl çalışır",
    "Karar ağacı ne işe yarar",
    "Decision tree algoritması nasıl kullanılır",

    # Naive Bayes
    "Naive Bayes nedir",
    "Bayes algoritması nasıl çalışır",
    "Naive Bayes nerelerde kullanılır",
    "Bayes sınıflandırması nedir",
    "Naive Bayes olasılık nasıl hesaplar",

    # SVM
    "SVM nedir",
    "Destek vektör makineleri nedir",
    "SVM nasıl çalışır",
    "Support vector machine nedir",
    "SVM ile sınıflandırma nasıl yapılır",

    # NumPy
    "NumPy nedir",
    "NumPy ne işe yarar",
    "Array nedir",
    "NumPy kütüphanesi nedir",
    "NumPy ile dizi nasıl oluşturulur",

    # Pandas
    "Pandas nedir",
    "DataFrame nedir",
    "Pandas ne işe yarar",
    "Excel verileri nasıl okunur",
    "Pandas ile veri analizi nasıl yapılır",

    # Matplotlib
    "Matplotlib nedir",
    "Python grafik nasıl çizilir",
    "Matplotlib ne işe yarar",
    "Grafik çizmek için hangi kütüphane kullanılır",
    "Python ile sütun grafiği nasıl yapılır",

    # NLP
    "NLP nedir",
    "Doğal dil işleme nedir",
    "NLP ne işe yarar",
    "Bilgisayarlar metinleri nasıl işler",
    "Doğal dil işleme nerelerde kullanılır",

    # TF-IDF
    "TF-IDF nedir",
    "TF-IDF ne işe yarar",
    "Metinler sayılara nasıl dönüştürülür",
    "TF-IDF nasıl çalışır",
    "TF-IDF ile metin nasıl sayısallaştırılır",

    # Metin sınıflandırma
    "Metin sınıflandırma nedir",
    "Duygu analizi nedir",
    "Yorumlar nasıl sınıflandırılır",
    "Spam mesaj nasıl tespit edilir",
    "Olumlu olumsuz yorum analizi nasıl yapılır"
]

# 2. SORULARIN ETİKETLERİ

etiketler = (
    ["python"] * 5 +
    ["degisken"] * 5 +
    ["liste"] * 5 +
    ["dongu"] * 5 +
    ["fonksiyon"] * 5 +
    ["yapay_zeka"] * 5 +
    ["makine_ogrenmesi"] * 5 +
    ["derin_ogrenme"] * 5 +
    ["knn"] * 5 +
    ["karar_agaci"] * 5 +
    ["naive_bayes"] * 5 +
    ["svm"] * 5 +
    ["numpy"] * 5 +
    ["pandas"] * 5 +
    ["matplotlib"] * 5 +
    ["nlp"] * 5 +
    ["tfidf"] * 5 +
    ["metin_siniflandirma"] * 5
)

# 3. KONULARA AİT CEVAPLAR

cevaplar = {
    "python": "Python, veri analizi, yapay zeka ve web geliştirmede kullanılan bir programlama dilidir.",

    "degisken": "Değişken, programda veri saklamak için kullanılır. Örnek: sayi = 10",

    "liste": "Liste, birden fazla veriyi bir arada tutar. Örnek: sayilar = [10, 20, 30]. Eleman eklemek için append() kullanılır.",

    "dongu": "Döngüler tekrarlayan işlemleri yapar. Python'da for ve while döngüleri bulunur.",

    "fonksiyon": "Fonksiyon, belirli bir görevi gerçekleştiren kod bloğudur. def ile tanımlanır. return ise fonksiyonun sonucunu döndürür.",

    "yapay_zeka": "Yapay zeka, makinelerin öğrenme, tahmin ve karar verme gibi görevleri gerçekleştirmesini sağlayan teknolojilerdir.",

    "makine_ogrenmesi": "Makine öğrenmesi, bilgisayarların verilerden örüntüler öğrenerek tahmin yapmasını sağlayan yapay zeka alanıdır.",

    "derin_ogrenme": "Derin öğrenme, çok katmanlı yapay sinir ağlarını kullanan bir makine öğrenmesi yaklaşımıdır.",

    "knn": "KNN, yeni bir veriyi en yakın k komşusuna bakarak sınıflandıran algoritmadır.",

    "karar_agaci": "Karar ağacı, verileri özelliklerine göre dallara ayırarak sınıflandırma veya tahmin yapan algoritmadır.",

    "naive_bayes": "Naive Bayes, Bayes teoremine dayalı olasılıksal bir sınıflandırma algoritmasıdır. Spam tespitinde kullanılabilir.",

    "svm": "SVM, sınıfları birbirinden ayıran uygun karar sınırını bulmaya çalışan makine öğrenmesi algoritmasıdır.",

    "numpy": "NumPy, sayısal hesaplamalar ve çok boyutlu diziler için kullanılan Python kütüphanesidir.",

    "pandas": "Pandas, tablolu verileri analiz etmek için kullanılır. DataFrame ve Series veri yapılarını sağlar.",

    "matplotlib": "Matplotlib, Python ile çizgi, sütun, pasta ve farklı grafik türleri oluşturmayı sağlar.",

    "nlp": "NLP yani doğal dil işleme, bilgisayarların insan dilindeki metinleri ve konuşmaları işlemesini sağlayan yapay zeka alanıdır.",

    "tfidf": "TF-IDF, bir kelimenin belgelerdeki önemini sayısal olarak ifade eden metin özellik çıkarma yöntemidir.",

    "metin_siniflandirma": "Metin sınıflandırma, yazılı metinleri önceden belirlenen kategorilere ayırma işlemidir. Örneğin olumlu veya olumsuz yorum tespiti."
}

# 4. METİNLERİ SAYISAL VERİLERE DÖNÜŞTÜRME

vec = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vec.fit_transform(sorular)

# 5. MODELİ EĞİTME

model = MultinomialNB(alpha=0.5)

model.fit(X, etiketler)

print("Model başarıyla eğitildi!")

# 6. CHATBOT BAŞLANGICI

print("================================")
print("   YAPAY ZEKA DERS ASISTANI")
print("================================")

isim = input("Adınız: ").strip()

if isim == "":
    isim = "Öğrenci"

print("\nMerhaba", isim)
print("Python ve yapay zeka konularında soru sorabilirsin.")
print("Çıkmak için q yaz.\n")

# 7. SOHBET DÖNGÜSÜ

while True:

    soru = input(isim + ": ").strip()

    # Çıkış
    if soru.lower() == "q":
        print("Bot: Görüşmek üzere!")
        break

    # Boş soru kontrolü
    if soru == "":
        print("Bot: Lütfen bir soru yaz.")
        continue

    # Soruyu sayısal verilere dönüştür
    yeni_soru = vec.transform([soru])

    # Soruda tanınan kelime yoksa
    if yeni_soru.nnz == 0:
        print("Bot: Bu soruyla ilgili bilgim yok.")
        print("Bot: Lütfen dersle ilgili başka bir soru sor.\n")
        continue

    # Konuyu tahmin et
    konu = model.predict(yeni_soru)[0]

    # Tahmin olasılıklarını hesapla
    olasiliklar = model.predict_proba(yeni_soru)[0]

    # En yüksek olasılığı bul
    en_yuksek = max(olasiliklar)

    # Cevabı ekrana yazdır
    print("\nBot:", cevaplar[konu])

    print("Tahmin edilen konu:", konu)
    print("Tahmin skoru:", round(en_yuksek, 2))
    print()
