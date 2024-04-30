from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle


# Create a Flask app
app = Flask(__name__)

with open('svc_model.pkl', 'rb') as file:
    svc_model = pickle.load(file)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define a route for the homepage
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        if uploaded_file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            data=df.values.tolist()
            return make_predictions(df, data)
            # return render_template('display_data.html', data=df.values.tolist())
        else:
            return 'Error: Please upload an Excel file!'
    else:
        return 'Error: No file uploaded!'

def update_labels(cols):
    outcome = cols[0]
    if outcome == 'attack':
        return 1
    else:
        return 0

def convert_to_list(lst):
    return [[sublist[0].tolist(), sublist[1][0].tolist()] for sublist in lst]

# Use Sample Data to make Predictions
def make_predictions(df, data):
    predictions = []
    image_files = []
    for i in range(len(df)):
        image_file = make_graph(df.iloc[0:i+1], image_files)  # Generate the graph image
        image_files.append(image_file)
    X = scaler.fit_transform(df)
    # print(X)
    #print(df.columns)  # No need to drop columns here
    predictions = svc_model.predict(X)
    confidence = svc_model.predict_proba(X)
    # Convert predictions to Python int
    print('predictions:', str(predictions))
    print('confidence:', str(confidence))
    confidence_percent = [[prob * 100 for prob in sample] for sample in confidence]
    print('confidence:', str(confidence_percent))
    image_files = [f'/static/{image}' for image in image_files]
    predictions_json = [[prediction.tolist(), confidence.tolist()] for prediction, confidence in zip(predictions, confidence)]
    return render_template('display_data.html', predictions=predictions_json, data=data, images = image_files)#predictions is a nested list where the inner lists represent model's prediction at that point, as well as the model's confidence in that prediction.

def make_graph(values, image_files):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axes = axs.ravel()
    features = ['shunt_voltage', 'bus_voltage_V', 'current_mA', 'power_mW']
    x_values = np.arange(len(values)) * 2  # Generate x-values as multiples of 2
    
    # Calculate the overall minimum and maximum values of each feature across all iterations
    min_values = values[features].min().min()
    max_values = values[features].max().max()
    y_range = max_values - min_values
    
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.plot(x_values, values[feature], marker='o')  # Plot only the latest data points
        ax.set_title(feature, fontsize=30)
        ax.set_xlabel('Time', fontsize=20)  # Set x-axis label and adjust font size
        ax.set_ylabel(feature.capitalize(), fontsize=20)  # Set y-axis label and adjust font size
        
        # Set fixed limits for the axes based on the overall range, with smaller margins
        ax.set_xlim(0, len(values) * 2)  # Set x-axis limits based on the length of the data
        ax.set_ylim(min_values - 0.05 * y_range, max_values + 0.05 * y_range)  # Set y-axis limits based on the overall range with smaller margins
        
        # Adjust the font size and style of tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=20)  

    plt.tight_layout()
    image_file = f'graph_{len(os.listdir("static"))}.png'
    image_path = os.path.join('static', image_file)
    plt.savefig(image_path)
    plt.close()
    return image_file
#creates 4 lineplots, (2,2) arragenment, of each feature used. It saves those graphs as graph.png and rewrites over that image with every iteration.

    # Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port = 15000) 