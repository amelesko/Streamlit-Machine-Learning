'''
Uses multiple ML classifiers to determine if mushrooms are poisonous.

Data is the mushroom data set from UCI's Machine Learning Repository.

Created as part of the Coursera project "Build a Machine Learning Web
    App with Streamlit and Python".

Created by: Alex Melesko
Date: 1/13/2022
'''

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def main():
    # Set up titles
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    # Loads specific dataset, fits label encoder, returns encoded labels
    # and caches
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    # Splits data into independent and dependent data, as well as
    # training and test sets, and caches
    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # Plots metrics if the user wants to
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()

    # Load and split up data
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    # Set up the classification names
    class_names = ['edible', 'poisonous']

    # Let the user choose the type of classifier
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        # Choose the hyperparameters
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernal Coefficient)", ("scale", "auto"), key='gamma')

        # Choose the metrics to plot
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            # Return the mean accuracy on the given test data and labels
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            # The precision is the ratio tp / (tp + fp)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            # The recall is the ratio tp / (tp + fn)
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        # Choose the hyperparameters
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        
        # Choose the metrics to plot
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            # Return the mean accuracy on the given test data and labels
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            # The precision is the ratio tp / (tp + fp)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            # The recall is the ratio tp / (tp + fn)
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        # Choose the hyperparameters
        n_estimators = st.sidebar.number_input("The number of tress in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees?", ("True", "False"), key='bootstrap')

        # Choose the metrics to plot
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            # Return the mean accuracy on the given test data and labels
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            # The precision is the ratio tp / (tp + fp)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            # The recall is the ratio tp / (tp + fn)
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    # Show raw data if the user likes
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set for Classification")
        st.write(df)

if __name__ == '__main__':
    main()


