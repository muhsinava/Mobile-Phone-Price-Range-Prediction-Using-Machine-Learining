from flask import Flask, request, render_template
import pickle

import pandas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/input')
def home():
    return render_template('input_form.html')

@app.route('/result', methods=['POST'])
def result():
     # Collect and validate the input
    
    battery_power = int(request.form.get('battery_power'))  # Convert to integer
    int_memory = int(request.form.get('int_memory'))  # Assuming it's encoded as an integer
    mobile_wt = int(request.form.get('mobile_wt'))  # Assuming binary (1 for Yes, 0 for No)
    n_cores = int(request.form.get('n_cores'))  # Convert to float
    px_height = int(request.form.get('px_height'))  # Convert to float
    px_width = int(request.form.get('px_width'))  # Convert to float
    ram = int(request.form.get('ram'))  # Convert to float
    

      # Create input DataFrame
    input_features = [[battery_power, int_memory, mobile_wt, n_cores, px_height, px_width, ram]]
    input_df = pandas.DataFrame(input_features, columns=[
            'battery_power', 'int_memory', 'mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram'
        ])

    
        # Load the scaler and scale the input data
    with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    input_scaled = scaler.transform(input_df)

        # Load the model and make a prediction
    with open('svm_model.pkl', 'rb') as model_file:
            svm = pickle.load(model_file)
            prediction = svm.predict(input_scaled)
            result = prediction[0]

    return render_template('result.html', res=result)
   

if __name__ == '__main__':
    app.run(debug=True)
