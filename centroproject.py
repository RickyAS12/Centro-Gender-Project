import streamlit as st
import pandas as pd
#import openpyxl
import joblib

st.info("Keakuratan model machine learning berada di 93,6%. Diperlukan bahan training tambahan dan bantuan human checking untuk memastikan keakuratan maksimal.")

# Load the model from the file
clf = joblib.load('classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("GENDER CHECKING")
st.header("Sampling nama")
test_name = st.text_input("Masukkan nama yang ingin disampling :")
if test_name != "" :
    jk_label_for_test_predict = {1:"Male", 0:"Female"}

    test_predict = vectorizer.transform([test_name])
    res = clf.predict(test_predict)
    if test_name == 'NULL':
        st.write('Female')
    else :
        st.write(jk_label_for_test_predict[int(res)])

st.header("Upload File")
uploaded_file = st.file_uploader("Input File:")
if uploaded_file is not None:
    df1=pd.read_excel(uploaded_file)
    names = names = df1['FullName'].tolist()
    # Transform all names at once using the vectorizer
    names_transformed = vectorizer.transform(names)
    # Predict gender for all names at once
    predictions = clf.predict(names_transformed)
    # Map predictions to the gender labels
    jk_label = {1: "Male", 0: "Female"}
    df1['Gender ML'] = [jk_label[int(pred)] for pred in predictions]
    df1['Gender ML'] = df1['Gender ML'].where(df1['FullName'] != 'NULL', 'Female')
    st.write("Tampilan Tabel Keseluruhan :")
    st.dataframe(df1)
    
    # Mapping
    gender_mapping = {'Male': 'Male', 'Female': 'Female'}
    # Add a column for gender comparison
    df1['Mapped_Gender'] = df1['Gender'].map(gender_mapping)
    # Check for mismatches
    df1['Gender_Mismatch'] = df1['Mapped_Gender'] != df1['Gender ML']
    # Filter mismatched rows
    mismatch_df = df1[df1['Gender_Mismatch'] == True]
    # Display only mismatched data in a separate table
    if not mismatch_df.empty:
        st.warning(f"Terdapat {mismatch_df.shape[0]} input gender yang berbeda.")
        st.write("Tampilan Tabel Gender yang Berbeda :")
        st.dataframe(mismatch_df)  # Display only the mismatched rows
    else:
        st.success("Semua Gender di tabel sudah benar")  
  
