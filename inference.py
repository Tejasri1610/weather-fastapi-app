import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load model + class labels
model = load_model("weights/3dcnn_model.h5")
class_labels = np.load("weights/class_labels_3dcnn.npy", allow_pickle=True)

def preprocess_input(lat, lon, state_code, month):
    # Arrange input to match training: [lat, lon, state, month]
    x = np.array([[lat, lon, state_code, month]])

    # Pad and reshape to (1, 2, 2, 2, 1)
    padded = np.hstack((x, np.zeros((1, 4))))  # total 8 features
    reshaped = padded.reshape((1, 2, 2, 2, 1))
    return reshaped

def predict_event_type(lat, lon, state_code, month):
    input_tensor = preprocess_input(lat, lon, state_code, month)
    preds = model.predict(input_tensor)
    predicted_class_index = np.argmax(preds)
    predicted_label = class_labels[predicted_class_index]
    return str(predicted_label)
