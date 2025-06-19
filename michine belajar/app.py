from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# Inisialisasi Flask app
app = Flask(__name__)

# Load model dan data
model_path = os.path.join('model_cpu.pkl')
data_path = os.path.join('semua_data_yang_dilearning.csv')

model = pickle.load(open(model_path, 'rb'))
df_all = pd.read_csv(data_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cores = int(request.form['cores'])
        threads = int(request.form['threads'])
        base_clock = float(request.form['base_clock'])
        tdp = float(request.form['tdp'])

        input_features = np.array([[cores, threads, base_clock, tdp]])
        prediction = model.predict(input_features)[0]

        # Filter data mirip (toleransi untuk float)
        matches = df_all[
            (df_all['Cores'] == cores) &
            (df_all['Threads'] == threads) &
            (abs(df_all['Base Clock'] - base_clock) <= 0.2) &  # toleransi 0.2 GHz
            (abs(df_all['TDP'] - tdp) <= 10)                   # toleransi 10 watt
        ]

        # Kirim hasil prediksi + semua prosesor yang mirip
        return render_template('index.html',
                               prediction=prediction,
                               cores=cores,
                               threads=threads,
                               base_clock=base_clock,
                               tdp=tdp,
                               matched_data=matches.to_dict(orient='records'))

    except Exception as e:
        return render_template('index.html', prediction=f"Terjadi kesalahan: {e}")


@app.route('/data')
def data():
    return render_template('data.html', tables=[df_all.to_html(classes='data', header=True, index=False)])

if __name__ == '__main__':
    app.run(debug=True)
