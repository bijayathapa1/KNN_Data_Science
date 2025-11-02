import KNN_model
import joblib

def predict_breast_cancer(input_data):
    # Load the trained model, scaler, and label encoder
    knn_model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('Breast_cancer_scaler.joblib')
    label_encoder = joblib.load('Breast_cancer_label_encoder.joblib')
    
    # Preprocess the input data
    input_data_scaled = scaler.transform([input_data])
    
    # Make prediction
    prediction = knn_model.predict(input_data_scaled)
    
    # Decode the prediction back to original label
    prediction_label = label_encoder.inverse_transform(prediction)
    
    return prediction_label[0]