from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import pickle  # Assuming you've trained your model and saved it as a pickle file

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/", methods=["GET"])
def helloworld():
    return "Hello World"

@app.route("/predict", methods=["POST"])
def predict_house_price():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Extract input values
        avg_income = float(data['avg_income'])
        house_age = float(data['house_age'])
        num_rooms = float(data['num_rooms'])
        num_bedrooms = float(data['num_bedrooms'])
        population = float(data['population'])

        # Logging the inputs for debugging
        print(f"avg_income: {avg_income}, house_age: {house_age}, num_rooms: {num_rooms}, num_bedrooms: {num_bedrooms}, population: {population}")

        # Prepare the input for the model
        input_features = np.array([[avg_income, house_age, num_rooms, num_bedrooms, population]])

        # Make a prediction using the loaded model
        prediction = model.predict(input_features)

        # Convert NumPy array to a float
        predicted_value = float(prediction[0])

        # Logging the prediction
        print(f"Prediction: {predicted_value}")

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': predicted_value
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Logging the error for debugging
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use the port from the environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
