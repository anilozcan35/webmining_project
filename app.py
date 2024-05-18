import os
import warnings

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.models import (dt_algoritm, k_means, knn_algoritm, nb_algorithm,
                        regression_algorithm)
from src.prediction import get_prediction_records
from src.preprocess import preprocess
from src.streamlit_functions import (data_metadata, data_preview, data_profiling, data_profilingA)
# from src.utils import download_dataset_from_kaggle
# from st_pages import show_pages_from_config

warnings.filterwarnings("ignore")


def home():
    """Home Page of Streamlit UI"""
    st.title('Web Mining Project')
    st.subheader('Bu uygulama Web Mining (BLM-5121) Projesi kapsamında ML Algoritmaları için geliştirilmiştir.')
    st.markdown('**1. Project Proposal:** Proje önerisi ve proje hakkında bilgi alabilirsiniz.')
    st.markdown('**2. Algorithm Design & Modelling:** Algoritmalar hakkında bilgi alabilirsiniz.')
    st.markdown('**3. Project System Design:** Proje aşamaları ve sistem tasarımı hakkında bilgi alabilirsiniz.')
    st.markdown('**4. Dataset Info:** Veri seti önizlemesi yapabilirsiniz. Veri seti hakkında bilgi alabilirsiniz.')
    st.markdown('**5. Multi Class Classification:** Çoklu sınıflandırma uygulamasıdır. Veri seti üzerinde çoklu sınıflandırma yapabilirsiniz.')
    st.markdown('**6. Regression:** Regresyon uygulamasıdır. Veri seti üzerinde regresyon analizi yapabilirsiniz.')
    st.markdown('**7. Clustering:** Kümeleme uygulamasıdır. Veri seti üzerinde kümeleme analizi yapabilirsiniz.')
    st.markdown('**8. App Info. & Credits:** Bu projede kullanılan Framework ve Libraryleri içermektedir.')


def proposal():
    """Project Proposal Page"""
    with open(file="ProjectProposal.md", encoding="utf8") as p:
        st.markdown(p.read())


def algorithm():
    """Algorithm Design & Modelling Page"""
    st.title('Algorithm Design & Modelling')
    st.image(image="./pipeline/AlgorithmDesign&Modelling.png",
             caption="Algorithm Design & Modelling",
             width=200,
             use_column_width="auto"
             )


def pipeline():
    """Project System Design Page"""
    st.title('Project System Design')
    st.image(image="./pipeline/SystemDesign.jpg",
             caption="Project System Design",
            #  width=200, 
             use_column_width="auto"
             )


def data():
    """Dataset Information Page"""
    st.title('Dataset Exploratory Data Analysis(EDA)')
    tab1, tab2, tab3, tab4 = st.tabs(["Meta Data", "Preview", "Profile(Raw Data)", "Profile(Preprocess Data)"])

    with tab1:
        st.image(image="https://storage.googleapis.com/kaggle-datasets-images/1732554/2832282/1be2ae7e0f1bc3983e65c76bfe3a436e/dataset-cover.jpg?t=2021-11-20-09-31-54",
                 caption="Body Performance Dataset from Kaggle",
                 width=200,
                 use_column_width="auto"
                 )
        st.header("Meta Data")
        data_metadata(file_path=DATA_FILE)
        # st.page_link(page="http://www.google.com", label="Dataset Url: Kaggle", icon="🌎")

    with tab2:
        st.header("Data Preview")
        data_preview(file_path=DATA_FILE)

    with tab3:
        st.header("Raw Data Profiling")
        with open(file="data/profiling/RawDataProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)

    with tab4:
        st.header("Preprocess Data Profiling")
        with open(file="data/profiling/PreprocessDataProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)


def classification():
    """Multi Class Classification Page"""
    st.title('Multi Class Classification Models')
    tab1, tab2, tab3 = st.tabs(["Decision Tree (DT)", "K Nearest Neighbor (KNN)", "Naive Bayes"])

    with tab1:
        st.header("Decision Tree (DT) Algorithm")
        # st.write('Decision Tree işlemi yapılacak.')
        tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(["Training", "Model Charts", "Classification Report", "Prediction"])

        with tab1_1:
            st.header("Decision Tree Training Component")
            dt_train_button = st.button("Train DT Model ")
            if dt_train_button:
                classification_report, cm = dt_algoritm.dt_train(model_name="decision_tree")
                st.session_state["classification_report"] = classification_report
                st.session_state["cm"] = cm
                st.dataframe(classification_report)

        with tab1_2:
            st.header("Decision Tree (DT) Model Charts")
            # st.pyplot(cm)
            st.image(image="./charts/dt_confusion_matrix.png",
                     caption="Confusion Matrix of Decision Tree Model",
                     # width=200, use_column_width="auto"
                     )

        with tab1_3:
            st.header("Decision Tree (DT) Classification Report")
            # st.dataframe(classification_report)
            st.dataframe(pd.read_csv("charts/dt_classification_report.csv"))

        with tab1_4:
            st.header("Decision Tree (DT) Prediction")
            prediction_record = get_prediction_records(key_start=0)
            print(prediction_record)
            predict_clicked = st.button("Predict", key= 100)
            if predict_clicked:
                pred = dt_algoritm.dt_prediction(prediction_record)
                st.header("Prediction")
                st.write(pred)

    with tab2:
        st.header("K Nearest Neighbor (KNN)")
        tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs(["Training", "Model Charts", "Classification Report", "Prediction"])

        with tab2_1:
            st.header("K Nearest Neighbor Training Component")
            knn_train_button = st.button("Train KNN Model")
            selected_n_neighbor = st.slider("Set Neighbbor Count", min_value=2, max_value=6, value=4)
            if knn_train_button:
                classification_report, cm = knn_algoritm.knn_train(model_name="knn", k=selected_n_neighbor)
                st.session_state["classification_report"] = classification_report
                st.session_state["cm"] = cm
                st.dataframe(classification_report)

        with tab2_3:
            st.header("K Nearest Neighbor Prediction")
            prediction_record = get_prediction_records(key_start=11)
            print(prediction_record)
            knn_predict_clicked = st.button("Predict", key=101)

            if knn_predict_clicked:
                pred = knn_algoritm.knn_prediction(prediction_record)
                st.header("Prediction")
                st.write(pred)

        with tab2_2:
            st.header("K Nearest Neighbor Model Charts")
            # st.pyplot(st.session_state["cm"])
            st.image(image="./charts/knn_confusion_matrix.png",
                     caption="Confusion Matrix of Decision Tree Model",
                     # width=200, use_column_width="auto"
                     )

        with tab2_4:
            st.header("K Nearest Neighbor Other")
            # st.dataframe(classification_report)
            st.dataframe(pd.read_csv("charts/knn_classification_report.csv"))

    with tab3:
        st.header("Naive Bayes (NB)")
        tab3_1, tab3_2, tab3_3, tab3_4 = st.tabs(["Training", "Model Charts", "Classification Report", "Prediction"])

        with tab3_1:
            st.header("Naive Bayes Training Component")
            nb_train_button = st.button("Train NB Model", key=103)
            if nb_train_button:
                classification_report, cm = nb_algorithm.nb_train(model_name="naive_bayes")
                st.session_state["classification_report"] = classification_report
                st.session_state["cm"] = cm
                st.dataframe(classification_report)

        with tab3_2:
            st.header("Naive Bayes Model Charts")
            # st.pyplot(cm)
            st.image(image="./charts/nb_confusion_matrix.png",
                     caption="Confusion Matrix of Decision Tree Model",
                     # width=200, use_column_width="auto"
                     )

        with tab3_3:
            st.header("Naive Bayes Prediction")
            # st.write("Naive Bayes Prediction Process Will Be Done.")
            prediction_record = get_prediction_records(key_start=22)
            print(prediction_record)
            nb_predict_clicked = st.button("Predict", key=104)

            if nb_predict_clicked:
                pred = nb_algorithm.nb_prediction(prediction_record)
                st.header("Prediction")
                st.write(pred)

        with tab3_4:
            st.header("Other")
            # st.dataframe(classification_report)
            st.dataframe(pd.read_csv("charts/nb_classification_report.csv"))


def regression():
    """Regression Modelling Page"""
    st.title('Regression Model')
    tab1, tab2, tab3 = st.tabs(["Training", "Model Charts", "Prediction"])

    with tab1:
        st.header("Regression Model Training Component")
        rg_button_train = st.button("Train Regression Model", key=107)
        if rg_button_train:
            mse, best_params, best_model, fig = regression_algorithm.regression_train()
            bp = pd.DataFrame.from_dict(best_params, orient="index")
            bp.to_csv("charts/lr_best_params.csv", index=False)
            # st.session_state["mse"] = mse
            # st.session_state["best_params"] = bp.T
            # st.session_state["best_model"] = best_model
            # st.session_state["fig"] = fig
            st.dataframe(mse)
            st.dataframe(bp.T)
            st.dataframe(best_model)


    with tab2:
        st.header("Regression Model Chart")
        # st.pyplot(fig)
        st.image(image="./charts/lr_chart.png",
                    caption="Confusion Matrix of Regression Model",
                    # width=200, use_column_width="auto"
                    )        
        st.subheader("Output Of GridSearch Best Parameters:")
        
        st.dataframe(pd.read_csv("charts/lr_best_params.csv"))

    with tab3:
        st.header("Regression Model Prediction")
        prediction_record = get_prediction_records(key_start=44)
        reg_predict_button = st.button("Predict", key=105)

        if reg_predict_button:
            pred = regression_algorithm.regression_predict(prediction_record)
            st.header("Predicted Body Score Of Given Record is:")
            st.write(pred)


def clustering():
    """Clustering Page"""
    st.title('Clustering Model')
    tab1, tab2, tab3, tab4 = st.tabs(["Train", "Elbow Chart", "Cluster Chart", "Prediction"])

    with tab1:
        st.header("Training Component")

        st.write('Selection Training Param')

        max_iter = st.selectbox("Max Iter", (300, 350, 400))

        selected_n_cluster = st.slider("Set K", min_value=2, max_value=6, value=4)
        show_clicked = st.button("Train K-Means", key=106)
        if show_clicked:
            fig_elbow, fig_clusters = k_means.k_means_train(selected_n_cluster=selected_n_cluster, max_iter=max_iter)
            st.session_state["fig_elbow"] = fig_elbow
            st.session_state["fig_cluster"] = fig_clusters

    with tab2:
        st.subheader("Elbow Graph")
        st.write('K Params Optimal for 4')
        # st.pyplot(fig_elbow)
        st.image(image="./charts/kmeans_elbow.png",
                 caption="Elbow Graph of K-Means Model",
                 # width=200, use_column_width="auto"
                 )

    with tab3:
        st.subheader("Clusters")
        # st.pyplot(fig_clusters)
        st.image(image="./charts/kmeans_cluster.png",
                 caption="Clusters of K-Means Model",
                 # width=200, use_column_width="auto"
                 )

    with tab4:
        st.header("Prediction")
        prediction_record = get_prediction_records(key_start=33)
        print(prediction_record)
        cluster_predict = st.button("Predict", key=105)

        if cluster_predict:
            fig = k_means.k_means_predict(prediction_record, selected_n_cluster)
            st.header("Prediction of Point")
            st.pyplot(fig)


def app_credits():
    """App Info. & Credits Page"""
    st.title('App Info. & Credits')
    st.markdown('**Programming Language:** Python 3.12')
    st.markdown('**Libraries & Frameworks:** Pandas, Scikit-learn, Numpy, Matplotlib, Seaborn, Plotly, Yellowbrick')
    st.markdown('**UI:** [Streamlit](https://streamlit.io/)')
    st.markdown('**Dev. Tools:** Git, Docker')
    st.markdown('**Data Source:** [Kaggle](https://www.kaggle.com/kukuroo3/body-performance-data)')
    st.markdown('**Github Repo:** [Web Mining Project](https://github.com/)')
    st.markdown('**Dash Platform:** [Streamlit Community Cloud](https://streamlit.io/cloud)')
    st.markdown('**Dash Url:** [StreamLit App](https://web-mining-project.streamlit.app/)')
    st.markdown('**Developed by:** [Metin Uslu](http://linkedin.com/in/metinuslu) & [Anıl Özcan](https://www.linkedin.com/in/anil-ozcan-6ba16b152/)')


def get_menu(user_name=None, user_password=None):
    """Streamlit UI Menu
    Params:
        user_name: str
        user_password: str
    """

    # st.sidebar.image("static/sidebar_logo.png")
    # , use_column_width=True
    st.sidebar.title('Web Mining Project')
    side_menu = {
        'Giriş': home,
        'Project Proposal': proposal,
        'Algorithm Design & Modelling': algorithm,
        'Project System Design': pipeline,
        'Dataset Meta Data & EDA': data,
        'Multi Class Classification Algorithms': classification,
        'Regression Algorithms': regression,
        'Clustering Algorithms': clustering,
        'App. Info. & Credits': app_credits
    }

    if st.session_state.get('login_success'):
        choice = st.sidebar.radio('Applications', list(side_menu.keys()))
        side_menu[choice]()
    else:
        with st.sidebar:
            with st.form(key='login_form'):
                st.title('Loging Page')
                username = st.text_input('User Name')
                password = st.text_input('Password', type='password')
                if st.form_submit_button('Login'):
                    if username == user_name and password == user_password:
                        st.session_state['login_success'] = True
                        st.success('Giriş başarılı, yönlendiriliyorsunuz...')
                        st.experimental_rerun()
                    else:
                        st.error('Kullanıcı adı veya şifre yanlış.')
                        st.session_state['login_success'] = False
    # show_pages_from_config()


if __name__ == "__main__":

    # Set Constants
    ROOT_PATH = os.getcwd()
    CFG_PATH = os.path.join(ROOT_PATH, 'cfg')
    # ENV_PATH = os.path.join(ROOT_PATH, '.env')
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
    PROFILLING_PATH = os.path.join(DATA_PATH, 'profiling')

    os.makedirs(CFG_PATH, exist_ok=True)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(PROFILLING_PATH, exist_ok=True)

    ENV_FILE = os.path.join(CFG_PATH, '.env')
    DATA_FILE = os.path.join(RAW_DATA_PATH, 'bodyPerformance.csv')
    PREPROCESS_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, 'preprocessed_data.csv')

    # Load Environment Variables
    load_dotenv(dotenv_path=ENV_FILE, encoding='utf-8', verbose=False)

    # Streamlit Secrets
    USER_NAME = os.environ.get("STREAMLIT_USER_NAME") if os.environ.get("STREAMLIT_USER_NAME") is not None else st.secrets["streamlit"]["STREAMLIT_USER_NAME"]
    USER_PASSWORD = os.environ.get("STREAMLIT_USER_PASSWORD") if os.environ.get("STREAMLIT_USER_PASSWORD") is not None else st.secrets["streamlit"]["STREAMLIT_USER_PASSWORD"]
    # print(USER_NAME, USER_PASSWORD)
    
    # Kaggle Secrets
    # KAGGLE_USER_NAME = os.environ.get("KAGGLE_USER_NAME") if os.environ.get("KAGGLE_USER_NAME") is not None else st.secrets["kaggle"]["KAGGLE_USER_NAME"]
    # KAGGLE_KEY = os.environ.get("KAGGLE_KEY") if os.environ.get("KAGGLE_KEY") is not None else st.secrets["kaggle"]["KAGGLE_KEY"]
    # print(KAGGLE_USER_NAME, KAGGLE_KEY)

    st.set_page_config(
        page_title="Web Mining Project UI ",
        page_icon=":gem:",
        layout="wide",
        # layout="centered",  
        initial_sidebar_state="expanded",
        # initial_sidebar_state="auto",
        # menu_items=None,
        menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
                    'Report a bug': "https://www.extremelycoolapp.com/bug",
                    'About': "# This is a header. This is an *extremely* cool app!"
                    })

    # Download Dataset from Kaggle
    # download_dataset_from_kaggle(user_name="kukuroo3", dataset_name="body-performance-data", path=RAW_DATA_PATH)
    # download_dataset_from_kaggle(kaggle_user_name=KAGGLE_USER_NAME, kaggle_key=KAGGLE_KEY, dataset_user_name="kukuroo3", dataset_name="body-performance-data", path=RAW_DATA_PATH)

    # df_data = data_load(file_name=DATA_FILE)
    df_preprocess_data, _ = preprocess(pred_mode=False, df=None)
    df_preprocess_data.to_csv(os.path.join(PREPROCESSED_DATA_PATH, "preprocessed_data.csv"), index=False)

    # Data Profiling
    data_profilingA(file_path=DATA_FILE, report_path=PROFILLING_PATH, minimal=False, report_file_name="RawDataProfilingReport")
    data_profilingA(file_path=PREPROCESS_DATA_FILE, report_path=PROFILLING_PATH, minimal=False, report_file_name="PreprocessDataProfilingReport")

    # Streamlit Menu
    get_menu(user_name=USER_NAME, user_password=USER_PASSWORD)
