from flask import Flask,request, render_template
import json
import joblib

model = joblib.load("model.pkl")
transformer = joblib.load("transformer.pkl")
app = Flask(__name__)

@app.route("/",methods=['GET',"POST"])
def home():
    return render_template("index.html")

@app.route("/predict",methods=['GET','POST'])
def main():
    data = request.form
    data = dict(data)
    temp_data = [data['cs'],data['geo'],data['gen'],data['age'],data['bal'],data['nop'],data['iam']]
    temp_data = transformer.transform([temp_data])
    data['Prediction_proba'] = model.predict_proba(temp_data).tolist()
    data['Prediction'] = model.predict(temp_data).tolist()
    return render_template("output.html",output=data)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")

    
# Input --> {"data":[{"cs":650,"geo":"France","gen":"Female","age":45,"bal":124450,"nop":5,"iam":0},
#{"cs":850,"geo":"Germany","gen":"Male","age":25,"bal":0,"nop":2,"iam":1}]}