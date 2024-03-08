from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictionPipeline, PredictionPipelineConfig, CustomData
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/prediction", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_education = request.form.get('parental_level_of_education')
        lunch = str(request.form.get('lunch'))
        test_preparation = request.form.get('test_preparation_course')
        reading_score = request.form.get('reading_score')
        writing_score = request.form.get('writing_score')

        # Error handling for missing fields or incorrect data types
        if not all([gender, ethnicity, parental_education, lunch, test_preparation, reading_score, writing_score]):
            return jsonify({"error": "Missing fields"}), 400

        try:
            reading_score = float(reading_score)
            writing_score = float(writing_score)
        except ValueError:
            return jsonify({"error": "Invalid score values"}), 400

        # Now you can create CustomData object and use it for prediction
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_preparation,
            reading_score=reading_score,
            writing_score=writing_score
        )
        data_df = data.get_data_as_df()
        print(data_df)
        prediction = PredictionPipeline(PredictionPipelineConfig()).predict(data_df)
        print(prediction)
        # return jsonify({"prediction": prediction})
        return render_template("index.html", prediction=prediction[0])
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
