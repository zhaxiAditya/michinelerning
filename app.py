import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan data
model = pickle.load(open('model_cpu.pkl', 'rb'))
df_all = pd.read_csv('semua_data_yang_dilearning.csv')

st.title("Prediksi Nama CPU Berdasarkan Spesifikasi")

# Form input
cores = st.number_input("Jumlah Cores", min_value=1, max_value=64, step=1)
threads = st.number_input("Jumlah Threads", min_value=1, max_value=128, step=1)
base_clock = st.number_input("Base Clock (GHz)", min_value=0.5, max_value=6.0, step=0.1, format="%.1f")
tdp = st.number_input("TDP (Watt)", min_value=1.0, max_value=500.0, step=1.0)

if st.button("Prediksi"):
    try:
        input_features = np.array([[cores, threads, base_clock, tdp]])
        prediction = model.predict(input_features)[0]
        st.success(f"Nama CPU diprediksi: **{prediction}**")

        # Filter data prosesor yang mirip
        matches = df_all[
            (df_all['Cores'] == cores) &
            (df_all['Threads'] == threads) &
            (abs(df_all['Base Clock'] - base_clock) <= 0.2) &
            (abs(df_all['TDP'] - tdp) <= 10)
        ]

        st.subheader("Prosesor yang Mirip:")
        st.dataframe(matches)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
