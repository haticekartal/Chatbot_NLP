
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

metinler = [
    "Python programlama dili öğreniyorum",
    "Python ile yazılım geliştiriyorum",
    "Bugün hava çok güzel",
    "Yazılım ve programlama öğrenmek istiyorum"
]

vec = TfidfVectorizer()

sayisal_veriler = vec.fit_transform(metinler)

benzerlik = cosine_similarity(sayisal_veriler)

print("Metinler arasındaki benzerlik:\n")
print(benzerlik)

print("\n1. ve 2. metin benzerliği:", benzerlik[0][1])
