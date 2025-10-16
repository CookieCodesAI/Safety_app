from src.preprocessing import structure_data
from src.model import make_model
from tensorflow.keras.models import load_model
from src.evaluate import evaluate

X_data, y_data = structure_data()
model, _,_,X_test, y_test, testing = make_model(X_data, y_data)

evaluate(model, X_test, y_test, testing)



