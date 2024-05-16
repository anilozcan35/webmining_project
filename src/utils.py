import os

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)


def download_dataset_from_kaggle_old(user_name: str, dataset_name: str, path: str):
    if not os.path.isfile(os.path.join(path, "bodyPerformance.csv")):
        os.makedirs(path, exist_ok=True)
        # Kaggle API objesini oluşturun
        api = KaggleApi()

        # API anahtarını yükleyin
        # api.set_config_value('username', 'asdasdasd')
        # api.set_config_value('key', 'asdasdasd')

        # api.set_config_value('username', st.secrets["kaggle"]["username"])
        # api.set_config_value('key', st.secrets["kaggle"]["key"])

        # API anahtarını yükleyin
        api.authenticate()

        # Veri setini indirin
        api.dataset_download_files(user_name + '/' + dataset_name, path=path, unzip=True)
    else:
        print("Dataset already exists!")


def download_dataset_from_kaggle(kaggle_user_name: str, kaggle_key: str, dataset_user_name: str, dataset_name: str, path: str):
    if not os.path.isfile(os.path.join(path, "bodyPerformance.csv")):
        os.makedirs(path, exist_ok=True)
        # Kaggle API objesini oluşturun
        api = KaggleApi()

        # API anahtarını yükleyin
        # api.set_config_value('username', st.secrets["kaggle"]["username"])
        # api.set_config_value('key', st.secrets["kaggle"]["key"])
        api.set_config_value('username', kaggle_user_name)
        api.set_config_value('key', kaggle_key)
        api.authenticate()

        # Veri setini indirin
        api.dataset_download_files(dataset_user_name + '/' + dataset_name, path=path, unzip=True)
    else:
        print("Dataset already exists!")


def data_load(file_path: str):
    """ Load data from file path
    :param file_path: str
    :return: pd.DataFrame
    """
    df_data = pd.read_csv(file_path)
    return df_data


def data_types(df: pd.DataFrame):
    """ Get data types of columns
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.dtypes
    df = df.to_frame(name='Data Types')
    df.index.name = 'Columns'
    return df


def missing_values(df: pd.DataFrame, threshold=0):
    """
        Calculate the missing (NaN) values and their percentage in the given DataFrame.

        Args:
        - df (pd.DataFrame): The DataFrame to be examined.
        - threshold (float, optional): Threshold percentage for missing values.
            Default is 0.

        Returns:
        - pd.DataFrame: A DataFrame containing the count and percentage of missing values.
            If a threshold is specified, it includes missing values with a percentage below
            the specified threshold.
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round((100 * df.isnull().sum()/df.isnull().count()).sort_values(ascending=False), 2)
    df_missing_data = pd.concat([total, percent], axis=1, keys=['Count', 'Percent']).sort_values(by='Percent', ascending=False)
    df_missing_data.index.name = 'Columns'

    if threshold == 0:
        return df_missing_data
    else:
        return df_missing_data[df_missing_data['Percent'] <= threshold]


def detect_cardinality(df: pd.DataFrame):
    """
        Detects the cardinality (number of unique values) for each feature in the given DataFrame.

        Args:
        - df (pd.DataFrame): The DataFrame for which cardinality is to be calculated.

        Returns:
        - pd.DataFrame: A DataFrame containing the cardinality count for each feature.
    """
    feature_cardinality = {}
        
    for column in df.columns:
        cardinality_value = df[column].nunique()
        feature_cardinality[column] = cardinality_value
    df_card_data = pd.DataFrame.from_dict(data=feature_cardinality, orient='index', columns=['Count'])
    df_card_data.index.name = 'Columns'

    return df_card_data


def evaluate_classification(y_true, y_pred):
    """
        Calculate Classification Metrics
        
        Args:
        y_true: True Value
        y_pred: Predict Value
        
        Returns:
        results: Dict Values of each metric 
    """
    results = {}

    # Accuracy
    results['Accuracy'] = accuracy_score(y_true, y_pred)

    # Precision
    results['Precision'] = precision_score(y_true, y_pred)

    # Recall
    results['Recall'] = recall_score(y_true, y_pred)

    # F1 Score
    results['F1 Score'] = f1_score(y_true, y_pred)

    # Confusion Matrix
    results['Confusion Matrix'] = confusion_matrix(y_true, y_pred)

    # fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # results['fpr'] = fpr
    # results['tpr'] = tpr
    # results['threshold'] = thresholds
    # results['Area Under Curve (AUC)'] = auc(fpr, tpr)

    return results