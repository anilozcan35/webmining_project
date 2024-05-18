import pandas as pd
import streamlit as st

from src import model, preprocess


def nb_train(model_name = "naive_bayes"):
        st.write("**Training Decision Tree Model**...")
        df = preprocess.get_data()
        df, arg = preprocess.preprocess(pred_mode=False, df=df)
        ct = model.ClassificationTask(dataframe=df, task_type="classification")

        classification_report, figure, params = ct.tune_and_predict_classification(model_name=model_name)
        # Convert the dictionary to a DataFrame
        df_report = pd.DataFrame(classification_report).transpose()
        # Save the DataFrame to a CSV file
        df_report.to_csv('charts/nb_classification_report.csv', index=False)

        figure.savefig("charts/nb_confusion_matrix.png", dpi=300)
        st.write("**Model Parameters:**", params[1])
        st.write("**Training Done, Check Tabs**...")
        return classification_report, figure


def nb_prediction(record_list):
    st.write("**Predicting with Decision Tree Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="naive_bayes")
    return predicted_class
