import pandas as pd
import streamlit as st

from src import model, preprocess


# Şimdilik params none bırakıyorum zaman kalırsa impelemte edilir.
def knn_train(model_name = "knn", k = 4):
        st.write("**Training KNN Model**...")
        df = preprocess.get_data()
        df, arg = preprocess.preprocess(pred_mode=False, df=df)
        ct = model.ClassificationTask(dataframe=df, task_type="classification", k=k)

        classification_report, figure, params = ct.tune_and_predict_classification(model_name=model_name)
        # Convert the dictionary to a DataFrame
        df_report = pd.DataFrame(classification_report).transpose()
        # Save the DataFrame to a CSV file
        df_report.to_csv('charts/knn_classification_report.csv', index=False)

        figure.savefig("charts/knn_confusion_matrix.png", dpi=300)
        params[2]['k'] = k
        st.write("**Model Parameters:**", params[2])
        st.write("**Training Done, Check Tabs**...")
        return classification_report, figure


def knn_prediction(record_list):
    st.write("**Predicting with KNN Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="knn")
    return predicted_class
