from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import ridge regressor model & Standard Scaler pickle file
ridge_model = pickle.load(open("models/ridgereg.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))


@app.route("/", methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        temperature_info = float(request.form.get("Temperature"))
        rh_info = float(request.form.get("RH"))
        ws_info = float(request.form.get("Ws"))
        rain_info = float(request.form.get("Rain"))
        ffmc_info = float(request.form.get("FFMC"))
        dmc_info = float(request.form.get("DMC"))
        isi_info = float(request.form.get("ISI"))
        classes_info = float(request.form.get("Classes"))
        region_info = float(request.form.get("Region"))
        
        new_data_scaled = standard_scaler.transform([[temperature_info,rh_info,ws_info,rain_info,ffmc_info,dmc_info,isi_info,classes_info,region_info]])
        output = ridge_model.predict(new_data_scaled)

        return render_template("home.html",result = output[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(debug=True)
