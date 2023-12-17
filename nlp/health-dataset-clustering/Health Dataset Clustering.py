# -*- coding: utf-8 -*-
"""Health Dataset Clustering.ipynb
"""

!pip install Sastrawi --upgrade
!pip install nltk

import nltk
nltk.download('punkt')

# Upload data
from google.colab import files
uploaded = files.upload()

#import library
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#import dataset
df = pd.read_excel(r'Health Dataset.xlsx')
print(df.head())
print("")

#CASE FOLDING
df['Teks'] = df['Teks'].str.lower()
print("Case Folding".center(60,"─"))
print(df.head(5))
print("")

def hapus_tweet_special(text):
    #hapus tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    #hapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    #hapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #hapus incomplete URL
    return text.replace("http://", " ").replace("https://", " ")

df['Teks'] = df['Teks'].apply(hapus_tweet_special)

#menghilangkan angka
def hapus_number(text):
    return  re.sub(r"\d+", "", text)
df['Teks'] = df['Teks'].apply(hapus_number)

#menghilangkan tanda baca
def hapus_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
df['Teks'] = df['Teks'].apply(hapus_punctuation)

#hapus whitespace leading & trailing
def hapus_whitespace(text):
    return text.strip()
df['Teks'] = df['Teks'].apply(hapus_whitespace)

#white space dobel --> 1 whitespace
def hapus_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
df['Teks'] = df['Teks'].apply(hapus_whitespace_multiple)

def hapus_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)
df['Teks'] = df['Teks'].apply(hapus_single_char)

print("Perbaikan Term".center(60,"─"))
print(df['Teks'].head(),"")

#TOKENIZING
def word_tokenize_wrapper(text):
    return word_tokenize(text)
df['tokens'] = df['Teks'].apply(word_tokenize_wrapper)

print("Hasil Tokenisasi".center(60,"─"))
print(df['tokens'].head())
print("")

#REMOVE STOPWORD
stop_factory = StopWordRemoverFactory()
print(stop_factory.get_stop_words())
#more_stopword = ['ia','bahwa','oleh','alo','haloo','yg','udah','karna']
defaultStopword = ['a', 'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'arti', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'b', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagainamakah', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasannya', 'bahwasanya', 'baik', 'baiklah', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa', 'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'c', 'cara', 'caranya', 'cukup', 'cukupkah', 'cukuplah', 'cuma', 'd', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu', 'e', 'empat', 'enak', 'enggak', 'enggaknya', 'entah', 'entahlah', 'f', 'g', 'guna', 'gunakan', 'h', 'hadap', 'hai', 'hal', 'halo', 'hallo', 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'helo', 'hello', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'i', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'j', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'k', 'kadar', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kali', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan', 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kena', 'kenapa', 'kepada', 'kepadanya', 'kerja', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khusus', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kok', 'kurang', 'l', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'laku', 'lalu', 'lama', 'lamanya', 'langsung', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lihat', 'lima', 'luar', 'm', 'macam', 'maka', 'makanya', 'makin', 'maksud', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'masuk', 'mata', 'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mohon', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'n', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nya', 'nyaris', 'nyata', 'nyatanya', 'o', 'oleh', 'olehnya', 'orang', 'p', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'q', 'r', 'rasa', 'rasanya', 'rupa', 'rupanya', 's', 'saat', 'saatnya', 'saja', 'sajalah', 'salam', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai', 'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'sangkut', 'satu', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekaranglah', 'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 't', 'tadi', 'tadinya', 'tahu', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya', 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'toh', 'tuju', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'u', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'v', 'w', 'waduh', 'wah', 'wahai', 'waktunya', 'walau', 'walaupun', 'wong', 'x', 'y', 'ya', 'yaitu', 'yakin', 'yakni', 'yang', 'z']
dataStopword = stop_factory.get_stop_words()+defaultStopword#+more_stopword
stopword = stop_factory.create_stop_word_remover()
def remove_stopwords(words):
    return [word for word in words if word not in dataStopword]
print("Tanpa Stopword".center(60,"─"))
df['tokens_WSW'] = df['tokens'].apply(remove_stopwords)
print(df['tokens_WSW'].head())
print("")

#STEMMING
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemmed_wrapper(term):
    return stemmer.stem(term)
term_dict = {}
for document in df['tokens_WSW']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
#print(len(term_dict))
for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    #print(term,":" ,term_dict[term])
#print(term_dict)

def get_stemmed_term(document):
    return [term_dict[term] for term in document]

print("Hasil Stemming".center(60,"─"))
df['tokens'] = df['tokens_WSW'].apply(get_stemmed_term)
print(df['tokens'])
print("")

def freqDist_wrapper(text):
    return FreqDist(text)
df['tokens_fdist'] = df['tokens'].apply(freqDist_wrapper)
print("Frekuensi Token".center(60,"─"))

print(df['tokens_fdist'].head().apply(lambda x : x.most_common()))

df.to_excel("Text_Preprocessing.xlsx")
Text_Preprocessing = pd.read_excel (r'Text_Preprocessing.xlsx')
print(Text_Preprocessing.head())

output = df['tokens'].str.join(" ")
print(output)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
vectorizer = TfidfVectorizer() #min_df = minimal muncul di df dokumen

X = vectorizer.fit_transform(output)
print(X.shape)
print(X)

Sum_of_squared_distances = []
K = range(1,31)
for k in K:
   km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=36)
   km = km.fit(X)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=25, init='k-means++', max_iter=200, n_init=10, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_

cluster_center_tf=kmeans.cluster_centers_
print(cluster_center_tf)

listTF2 = cluster_center_tf.tolist()
#print(listTF2)
dataTF2 = pd.DataFrame(listTF2)
dataTF2.to_excel("TF-IDF2.xlsx")

from sklearn.decomposition import PCA

# PCA untuk mereduksi data
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# assign clusters and pca vectors to our dataframe
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score

get_top_keywords(20)

# map clusters to appropriate labels
cluster_map = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3:"Cluster 3", 4:"Cluster 4", 5: "Cluster 5", 6:"Cluster 6", 7:"Cluster 7", 8:"Cluster 8", 9:"Cluster 9", 10:"Cluster 10", 11:"Cluster 11", 12:"Cluster 12", 13:"Cluster 13", 14:"Cluster 14", 15:"Cluster 15", 16:"Cluster 16", 17:"Cluster 17", 18:"Cluster 18", 19:"Cluster 19", 20:"Cluster 20", 21:"Cluster 21", 22:"Cluster 22", 23:"Cluster 23", 24:"Cluster 24", 25:"Cluster 25", 26:"Cluster 26"}
# apply mapping
df['cluster'] = df['cluster'].map(cluster_map)

# set image size
plt.figure(figsize=(20, 10))
# set a title
plt.title("TF-IDF + KMeans clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="icefire")
plt.show()

from collections import Counter
clustered_docs = kmeans.fit_predict(X)
def purity(labels, clustered):
    # find the set of cluster ids
    cluster_ids = set(clustered)
    N = len(clustered)
    majority_sum = 0
    for cl in cluster_ids:
        # for this cluster, we compute the frequencies of the different human labels we encounter
        # the result will be something like { 'camera':1, 'books':5, 'software':3 } etc.
        labels_cl = Counter(l for l, c in zip(labels, clustered) if c == cl)
        # we select the *highest* score and add it to the total sum
        majority_sum += max(labels_cl.values())
    # the purity score is the sum of majority counts divided by the total number of items
    #print("Majority",majority_sum, "N", N)
    return majority_sum / N

skor = purity(df["Teks"],clustered_docs)
print('Akurasi: {:.2f}%'.format(skor*100))