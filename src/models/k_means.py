import streamlit as st

from src import model, preprocess


def k_means_train(selected_n_cluster, max_iter):
    st.write("**Training k-Means Model**...")
    df, *args = preprocess.preprocess()
    clustering_task = model.ClusteringTask(df, n_cluster=selected_n_cluster, max_iter=max_iter)
    params = clustering_task.task_params[1]
    params['k'] = selected_n_cluster
    params['max_iter'] = max_iter

    fig_elbow = clustering_task.show_elbow()
    fig_elbow.savefig("charts/kmeans_elbow.png", dpi=300)

    fig_clusters = clustering_task.cluster_plots()
    fig_clusters.savefig("charts/kmeans_cluster.png", dpi=300)
    st.write("**Model Parameters:**", params)
    st.write("**Training k-Means Model Completed. Check Tabs**...")
    return fig_elbow, fig_clusters

def k_means_predict(record_list, selected_n_cluster):
    st.write("**Predicting k-Means Model**...")
    fig = model.ClusteringTask.predict(record_list=record_list, n_clusters=selected_n_cluster)
    # fig.savefig("charts/kmeans_cluster_predict.png", dpi=300)
    return fig
