import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder="template")
model = pickle.load(open('Breast_cancer.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['Age', 'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage','differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status','Progesterone Status', 'Regional Node Examined','Reginol Node Positive', 'Survival Months']
      

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 0:
      res_val = "Alive"
  else:
      res_val = "Dead"


  return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run()
