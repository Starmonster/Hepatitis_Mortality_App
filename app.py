#Import Core Packages
import streamlit as st

# EDA and Data packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import lime
import lime.lime_tabular


#ML packages
import sklearn

plt.style.use('fivethirtyeight')

# Import Utilities
import os
import joblib
import hashlib

from managed_db import *

def generate_hashes(password):

    """function to hash the password"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

#List of best features (see ipynb file for further information)
best_features = ['age', 'sex', 'steroid', 'antivirals', 'protime', 'sgot', 'bilirubin',
                 'alk_phosphate', 'albumin','spiders', 'histology', 'fatigue', 'ascites',
                 'varices']

gender_dictionary = {"male" : 1, "female" : 2}
feature_dictionary = {"No" : 1, "Yes" : 2}

#write a function to return a dict value when the key is passed as argument
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

#write a function to return a dict key when the key value is passed as argument
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key

#write a function to return a value 1/0 when yes / no passed as argument
def get_f_value(val):
    feature_dictionary = {"No":1, "Yes":2}
    for key, value in feature_dictionary.items():
        if val == key:
            return value

#load our pre-saved models
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """ Hepatitis Mortality Prediction App"""
    st.title("Hepatitis Disease Mortality Prediction App")

    menu = ["Home", "Login", "SignUp"]
    submenu = ["Plot", "Prediction", "Metrics" ]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text("What is Hepatitis")
    elif choice == "Login":

        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username,verify_hashes(password, hashed_pswd))

            # if password == "12345":
            if result:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Activity", submenu)
                if activity == "Plot":
                    st.subheader("Data Vis Plot")
                    df = pd.read_csv("clean_hep_dataset.csv")
                    st.dataframe(df)


                    fig, ax = plt.subplots()

                    # frequency distribution plot
                    freq_df = pd.read_csv("freq_df_hep_dataset.csv")
                    st.bar_chart(freq_df['count'])

                    #plot the live/die class count
                    df["class"].value_counts().plot(kind="bar")
                    plt.title("plot of target class counts in dataset", fontsize=14)
                    pos = (0,1)
                    labels = ('2: male', '1: female')
                    plt.xticks(pos, labels=labels, rotation=0)


                    st.pyplot(fig)

                    if st.checkbox("Area Chart"):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect("Choose a Feature", all_columns)
                        #create new dataframe with feature choices
                        new_df = df[feat_choices]
                        st.area_chart(new_df)

                elif activity == "Prediction":
                    st.subheader("Predictive Analytics")

                    #Create a form to enter in the patients details
                    st.subheader("Please enter your details in the below form")
                    age = st.number_input("Age", 7, 80)
                    sex = st.radio("Sex", tuple(gender_dictionary.keys()))
                    steroid = st.radio("Do you take steroids?", tuple(feature_dictionary.keys()))
                    anti_virals = st.radio("Do you take anti-virals?", tuple(feature_dictionary.keys()))
                    protime = st.number_input("Protime", 0.0, 100.0)
                    sgot = st.number_input("SGOT", 0.0, 648.0)
                    bilirubin = st.number_input("Bilirubin", 0.0, 8.0)
                    alk = st.number_input("Alk-Phosphate", 0.0, 295.0)
                    albumin = st.number_input("Albumin", 0.0, 6.4)
                    spiders = st.radio("Spiders", tuple(feature_dictionary.keys()))
                    histology = st.selectbox("Histology", tuple(feature_dictionary.keys()))
                    fatigue = st.radio("Do you have fatigue?", tuple(feature_dictionary.keys()))
                    ascites = st.selectbox("Ascites", tuple(feature_dictionary.keys()))
                    varices = st.selectbox("Varices", tuple(feature_dictionary.keys()))



                    #Create a list of all the details
                    feature_list = [age, get_value(sex, gender_dictionary), get_f_value(steroid),
                                    get_f_value(anti_virals), protime, sgot, bilirubin, alk,
                                    albumin, get_f_value(spiders), get_f_value(histology),
                                    get_f_value(fatigue), get_f_value(ascites), get_f_value(varices)]
                    #display the list in a summary
                    st.write(feature_list)

                    #Display a more informative version
                    st.text("the following list is in a different order to above list")
                    pretty_result = {"age":age, "sex":sex, "bilirubin":bilirubin,
                                     "alk_phosphate": alk, "SGOT": sgot, "albumin": albumin,
                                     "protime":protime, "steroid": steroid, "anti_virals":anti_virals,
                                     "spiders": spiders, "histology": histology, "varices": varices,
                                     "ascites": ascites, "fatigue": fatigue}
                    st.json(pretty_result)

                    #Reshape the details so they can be accepted by our ML models
                    single_sample = np.array(feature_list).reshape((1, -1))

                    #ML modelling of the entered details
                    model_choice = st.selectbox("Select Model", ["LR", "KNN", "Decision Tree"])
                    if st.button("Predict"):
                        if model_choice == "KNN":
                            loaded_model = load_model("knn_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                            st.write(prediction)
                            st.write(pred_prob)
                        elif model_choice == "Decision Tree":
                            loaded_model = load_model("dt_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                            st.write(prediction)
                            st.write(pred_prob)
                        elif model_choice == "LR":
                            loaded_model = load_model("logistic_regression_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                            st.write(prediction)
                            st.write(pred_prob)

                        st.write(prediction)
                        prediction_label = {"Die":1, "Live": 2}
                        fina_result = get_key(prediction, prediction_label)
                        st.write(fina_result)

                        if prediction == 1:
                            st.warning("Patient Dead")
                            pred_prob_score = {"Die %": pred_prob[0][0]*100, "Live %": pred_prob[0][1]*100}
                            st.subheader("Prediction Probability score using {}".format(model_choice))
                            st.write(pred_prob_score)

                        else:
                            st.success("Patient does not die")
                            pred_prob_score = {"Die %": pred_prob[0][0]*100, "live %":pred_prob[0][1]*100}
                            st.subheader("Prediction Probability score using {}".format(model_choice))
                            st.json(pred_prob_score)





                elif activity == "Metrics":
                    st.subheader("Metrics and Evaluation")

            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type="password")

        confirm_password = st.text_input("Confirm Password", type="password")
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")

        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login to get started")
            pass



if __name__ == '__main__':
    main()

