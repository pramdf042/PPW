import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  
from sklearn import tree

Data, Ekstraksi, lda, LDAkmeans, Model = st.tabs(['Data', 'Ekstraksi Fitur', 'LDA', 'Modelling'])

with Data :
   st.title("""UTS PPW A""")
   st.text('Pramudya Dwi Febrianto 200411100042')
   st.subheader('Deskripsi Data')
   st.text("""
            1) Judul
            2) Penulis
            3) Dosen Pembimbing 1
            4) Dosen Pembinbing 2
            5) Abstrak
            5) Label""")
   st.subheader('Data')
   data=pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Data%20Berlabel.csv')
   data

with Ekstraksi :

   st.subheader('Term Frequency (TF)')
   data_x = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
   data_x
   
#    st.subheader('Logarithm Frequency (Log-TF)')
#    log_tf = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/log_TF.csv')
#    log_tf
   
#    st.subheader('One Hot Encoder / Binary')
#    oht = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/OneHotEncoder.csv')
#    oht
   
#    st.subheader('TF-IDF')
#    tf_idf = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/TF-IDF.csv')
#    tf_idf

with lda:
        X = data_x.drop('Dokumen', axis=1)
        import numpy as np
        kelas_dataset = X['Label']
        kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in kelas_dataset]
        X['Label'] = kelas_dataset_binary

        k = 3
        alpha = 0.1
        beta = 0.2
        lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
        proporsi_topik_dokumen = lda.fit_transform(X)
        dokumen = data_x['Dokumen']
        output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
        output_proporsi_TD.insert(0,'Dokumen', dokumen)
        output_proporsi_TD.insert(len(output_proporsi_TD.columns),'Label', X['Label'])
        output_proporsi_TD

with LDAkmeans:
      # Mengambil fitur dari X
        X = output_proporsi_TD[['Topik 1', 'Topik 2', 'Topik 3']]

        # Inisialisasi model K-Means dengan jumlah cluster yang diinginkan
        n_clusters = 2  # Ganti sesuai dengan jumlah cluster yang Anda inginkan
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        # Melatih model K-Means dengan data X
        kmeans.fit(X)

        # Menambahkan label cluster ke data awal
        output_proporsi_TD['Cluster'] = kmeans.labels_

        # Hasil klastering
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        # Mengukur Inertia
        inertia = kmeans.inertia_
        st.write(f"Inertia: {inertia}")
        output_proporsi_TD
   
with Model :
    # if all :
        # Contoh data latih dan label
        X = output_proporsi_TD[['Topik 1', 'Topik 2', 'Topik 3']]
        y = output_proporsi_TD['Label']
        # Pisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        metode1 = KNeighborsClassifier(n_neighbors=3)
        metode1.fit(X_train, y_train)

        metode2 = GaussianNB()
        metode2.fit(X_train, y_train)

        metode3 = tree.DecisionTreeClassifier(criterion="gini")
        metode3.fit(X_train, y_train)

        st.write ("Pilih metode yang ingin anda gunakan :")
        met1 = st.checkbox("KNN")
        # if met1 :
        #     st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
        met2 = st.checkbox("Naive Bayes")
        # if met2 :
        #     st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
        met3 = st.checkbox("Decesion Tree")
        # if met3 :
            # st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
            # st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
        submit2 = st.button("Pilih")

        if submit2:      
            if met1 :
                st.write("Metode yang Anda gunakan Adalah KNN")
                st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
            elif met2 :
                st.write("Metode yang Anda gunakan Adalah Naive Bayes")
                st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
            elif met3 :
                st.write("Metode yang Anda gunakan Adalah Decesion Tree")
                st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
            else :
                st.write("Anda Belum Memilih Metode")
    # else:
    #     st.write("Anda Belum Menentukan Jumlah Topik di Menu LDA")
