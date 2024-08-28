from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle



model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods = ['GET', 'POST'])
def predict2():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']
    

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {
        1: "Rice",
        2: "Maize" ,3: "Jute" ,4: "Cotton" ,  5: "Coconut",    6: "Papaya", 7: "Orange",8: "Apple",9: "Muskmelon",
        10: "Watermelon",11: "Grapes",12: "Mango",13: "Banana",14: "Pomegranate",15: "Lentil",16: "Blackgram",
        17: "Mungbean",18: "Mothbeans",19: "Pigeonpeas",20: "Kidneybeans",21: "Chickpea",22: "Coffee",
            }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result=("{} is a Best crop to be Cultivated ".format(crop))
    else:
        result=("Sorry are not able to recommand a Proper Crop for this Environmnet")

    return render_template("index.html",result=result)
   
    
    

if __name__ =="__main__":
    app.run(debug=True)