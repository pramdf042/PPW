import streamlit as st
from streamlit_option_menu import option_menu
from sklearn import datasets
from sklearn. tree import DecisionTreeClassifier
import numpy as np
from sklearn import datasets
from math import e
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans



st.set_page_config(
    page_title="PPW",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQS91LtGD9MoCRfZde90o4A3sVKS3KmJ8hkUWsGMjDIIKH66h1C_2QyvEL-4EPoS1SyMeU&usqp=CAU",
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">UTS PPW A</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Mula'ab, S.Si., M.Kom.",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQS91LtGD9MoCRfZde90o4A3sVKS3KmJ8hkUWsGMjDIIKH66h1C_2QyvEL-4EPoS1SyMeU&usqp=CAU" width="150" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home", "prepocessing","LDA","K-means","model", "Implementation"], 
            icons=['house', 'bar-chart','check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQS91LtGD9MoCRfZde90o4A3sVKS3KmJ8hkUWsGMjDIIKH66h1C_2QyvEL-4EPoS1SyMeU&usqp=CAU" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    
    if selected =="prepocessing" :
        st.subheader('Term Frequency (TF)')
        tf = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
        tf
        
        # st.subheader('Logarithm Frequency (Log-TF)')
        # log_tf = pd.read_csv('')
        # log_tf
        
        # st.subheader('One Hot Encoder / Binary')
        # oht = pd.read_csv('')
        # oht
        
        # st.subheader('TF-IDF')
        # tf_idf = pd.read_csv('')
        # tf_idf


    if selected =="LDA" :
        data_x = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
        import numpy as np
        kelas_dataset = data_x['Label']

        # Ubah kelas menjadi 0 dan kelas B menjadi 1
        kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in kelas_dataset]

        # Contoh cetak hasilnya
        data_x['Label']=kelas_dataset_binary

        X = data_x.drop('Dokumen', axis=1)
        k = 3
        alpha = 0.1
        beta = 0.2

        lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
        proporsi_topik_dokumen = lda.fit_transform(X)

        dokumen = data_x['Dokumen']
        label= data_x['Label']
        output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
        output_proporsi_TD.insert(0,'Dokumen', dokumen)
        output_proporsi_TD.insert(len(output_proporsi_TD.columns),'Label', data_x['Label'])
        st.subheader('Proporsi Topik Dokumen')
        output_proporsi_TD

        # Output distribusi kata pada topik
        distribusi_kata_topik = pd.DataFrame(lda.components_)
        st.subheader('Distribusi Kata Topik')
        distribusi_kata_topik

    if selected =="K-means" :
        data_x = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
        import numpy as np
        kelas_dataset = data_x['Label']

        # Ubah kelas A menjadi 0 dan kelas B menjadi 1
        kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in kelas_dataset]

        # Contoh cetak hasilnya
        data_x['Label']=kelas_dataset_binary

        X = data_x.drop('Dokumen', axis=1)
        k = 3
        alpha = 0.1
        beta = 0.2

        lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
        proporsi_topik_dokumen = lda.fit_transform(X)

        dokumen = data_x['Dokumen']
        label= data_x['Label']
        output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
        output_proporsi_TD.insert(0,'Dokumen', dokumen)
        output_proporsi_TD.insert(len(output_proporsi_TD.columns),'Label', data_x['Label'])
        
        # output_proporsi_TD

        # Output distribusi kata pada topik
        distribusi_kata_topik = pd.DataFrame(lda.components_)
        
        # distribusi_kata_topik
        # Melakukan clustering dengan K-Means
        X_clustering = proporsi_topik_dokumen
        n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(X_clustering)

        # Menambahkan hasil clustering ke DataFrame
        output_proporsi_TD['Cluster'] = clusters

        # Menggabungkan DataFrame hasil LDA dan DataFrame hasil clustering
        output_final_df = pd.concat([output_proporsi_TD], axis=1)

        output_final_df

    if selected =="model" :
            data_x = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
            data_x = data_x.dropna(subset=['Dokumen'])  # Menghapus baris yang memiliki NaN di kolom 'Dokumen'
            import numpy as np
            kelas_dataset = data_x['Label']

            # Ubah kelas menjadi 0 dan kelas B menjadi 1
            kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in kelas_dataset]

            # Contoh cetak hasilnya
            data_x['Label']=kelas_dataset_binary

            X = data_x.drop('Dokumen', axis=1)
            k = 3
            alpha = 0.1
            beta = 0.2

            lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
            proporsi_topik_dokumen = lda.fit_transform(X)

            dokumen = data_x['Dokumen']
            label= data_x['Label']
            output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
            output_proporsi_TD.insert(0,'Dokumen', dokumen)
            output_proporsi_TD.insert(len(output_proporsi_TD.columns),'Label', data_x['Label'])
            # st.subheader('Proporsi Topik Dokumen')
            # output_proporsi_TD
        # Impor library yang diperlukan
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import accuracy_score

            # Bagian ini adalah contoh data Anda. Anda harus menggantinya dengan data nyata Anda.
            # X adalah matriks fitur, y adalah target/label
            # Memisahkan fitur dan label kelas target
            # Bagi data menjadi data pelatihan dan data pengujian
            X = data_x['Dokumen']
            label = data_x['Label']
            X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

            # Vektorisasi teks menggunakan TF-IDF
            tfidf_vectorizer = TfidfVectorizer()
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            # Buat model Decision Tree
            decision_tree_model = DecisionTreeClassifier()
            decision_tree_model.fit(X_train_tfidf, y_train)

            # Buat model K-Nearest Neighbors (KNN)
            knn_model = KNeighborsClassifier(n_neighbors=5)  # Anda dapat mengubah nilai n_neighbors sesuai kebutuhan
            knn_model.fit(X_train_tfidf, y_train)

            # Buat model Naive Bayes (Gaussian Naive Bayes)
            naive_bayes_model = MultinomialNB()
            naive_bayes_model.fit(X_train_tfidf, y_train)

            # Prediksi dengan model Decision Tree
            y_pred_decision_tree = decision_tree_model.predict(X_test_tfidf)

            # Prediksi dengan model KNN
            y_pred_knn = knn_model.predict(X_test_tfidf)

            # Prediksi dengan model Naive Bayes
            y_pred_naive_bayes = naive_bayes_model.predict(X_test_tfidf)

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
                    # Hitung akurasi model KNN dan konversi ke persentase
                    accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
                    st.write(f"Akurasi KNN: {accuracy_knn:.2f}%")
                elif met2 :
                    st.write("Metode yang Anda gunakan Adalah Naive Bayes")
                    accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes) * 100
                    st.write(f"Akurasi Naive Bayes: {accuracy_naive_bayes:.2f}%")
                elif met3 :
                    st.write("Metode yang Anda gunakan Adalah Decesion Tree")
                    accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree) * 100
                    st.write(f"Akurasi Decision Tree: {accuracy_decision_tree:.2f}%")
                else :
                    st.write("Anda Belum Memilih Metode")
    if selected == "Implementation":
        # from sklearn.datasets import load_breast_cancer
        # from sklearn.model_selection import train_test_split
        # from sklearn.metrics import accuracy_score


        # breast_cancer = load_breast_cancer()
        # X = breast_cancer.data
        # y = breast_cancer.target

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # X_train_subset = X_train[:,:5]
        # y_train_subset = y_train[:]


        # # Create and train AdaBoostClassifier
        # adaboost = AdaBoostClassifier(n_estimators=3, learning_rate=0.1)
        # adaboost.fit(X_train_subset, y_train_subset)

        # y_pred = adaboost.predict(X_test[:,:5])

        # accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)

        # with st.form("my_form"):
        #     st.subheader("Implementasi")
        #     mean_radius = st.number_input('Masukkan Mean radius')
        #     mean_tektstur = st.number_input('Masukkan Mean texture')
        #     mean_perimeter = st.number_input('Masukkan Mean perimeter')
        #     mean_area = st.number_input('Masukkan Mean area')
        #     mean_smoothness = st.number_input('Masukkan Mean smoothness')
        #     submit = st.form_submit_button("submit")
            
        #     if submit:
        #         st.subheader('Hasil Prediksi')
        #         inputs = np.array([mean_radius,mean_tektstur,mean_perimeter,mean_area,mean_smoothness])
        #         input_norm = np.array(inputs)
        #         input_pred = adaboost.predict(input_norm)
        #     # Menampilkan hasil prediksi
        #         if input_pred=='0':
        #             st.success('malignant')
        #         else :
        #             st.success('benign')

        import pandas as pd
        import numpy as np
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Baca data
        data_x = pd.read_csv('https://raw.githubusercontent.com/pramdf042/PPW/main/Term%20Frequensi%20Berlabel%20Final.csv')
        data_x = data_x.dropna(subset=['Dokumen'])  # Menghapus baris yang memiliki NaN di kolom 'Dokumen'

        # Ubah kelas A menjadi 0 dan kelas B menjadi 1
        kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in data_x['Label']]
        data_x['Label'] = kelas_dataset_binary

        # Bagi data menjadi data pelatihan dan data pengujian
        X = data_x['Dokumen']
        label = data_x['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

        # Vektorisasi teks menggunakan TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Latih model LDA
        k = 3
        alpha = 0.1
        beta = 0.2
        lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
        proporsi_topik_dokumen = lda.fit_transform(X_train_tfidf)

        import pickle
        with open('nb.pkl', 'rb') as file:
            nb = pickle.load(file)

        with st.form("my_form"):
            st.subheader("Implementasi")
            input_dokumen = st.text_input('Masukkan Abstrak')
            input_vector = tfidf_vectorizer.transform([input_dokumen])
            submit = st.form_submit_button("submit")
            # Prediksi proporsi topik menggunakan model LDA
            proporsi_topik = lda.transform(input_vector)[0]
            if submit:
                st.subheader('Hasil Prediksi')
                inputs = np.array([input_dokumen])
                input_norm = np.array(inputs)
                input_pred = nb.predict(input_vector)[0]
            # Menampilkan hasil prediksi
                if input_pred==0:
                    st.success('RPL')
                    st.write(proporsi_topik)
                else  :
                    st.success('KK')
                    st.write(proporsi_topik)
