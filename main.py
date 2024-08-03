import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import category_encoders as ce

app = Flask(__name__)

data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RandomForest_best.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    balcony = request.form.get('bal')
    sqft = request.form.get('sqft')
    print(location, bhk, bath, balcony, sqft)
    input = pd.DataFrame([[location,sqft,bath,balcony,bhk]],columns=['location', 'total_sqft', 'bath', 'balcony', 'bhk'])
    encoder = ce.BinaryEncoder(cols=['location'])
    input_encoded = encoder.fit_transform(input)
    prediction = pipe.predict(input_encoded)[0]*1e5

    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug=True, port=80)