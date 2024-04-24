from flask import Flask, render_template, request
import pandas as pd
import os
import time

# Create a Flask app
app = Flask(__name__)

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
            return render_template('display_data.html', data=df.values.tolist())
        else:
            return 'Error: Please upload an Excel file!'
    else:
        return 'Error: No file uploaded!'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port = 15000)