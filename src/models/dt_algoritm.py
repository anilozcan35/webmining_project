import pandas as pd
import streamlit as st

from src import model, preprocess


def dt_train(model_name = "decision_tree"):
        st.write("**Training Decision Tree Model**...")
        df = preprocess.get_data()
        df, arg = preprocess.preprocess(pred_mode=False, df=df)
        ct = model.ClassificationTask(dataframe=df, task_type="classification")
        
        classification_report, figure, params = ct.tune_and_predict_classification(model_name=model_name)
        # Convert the dictionary to a DataFrame
        df_report = pd.DataFrame(classification_report).transpose()
        # Save the DataFrame to a CSV file
        df_report.to_csv('charts/dt_classification_report.csv', index=False)

        figure.savefig("charts/dt_confusion_matrix.png", dpi=300)
        st.write("**Model Parameters:**", params[0])
        st.write("**Training Done, Check Tabs**...")
        return classification_report, figure


def dt_prediction(record_list):
    st.write("**Predicting with Decision Tree Model**...")
    predicted_class = model.ClassificationTask.predict(record_list=record_list, model_name="decision_tree")
    return predicted_class