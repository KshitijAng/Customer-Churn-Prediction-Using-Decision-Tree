from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = "decision_tree.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# FastAPI app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    Age: float
    NumOfProducts: float
    Balance: float

@app.post("/predict")
def predict(data: InputData):
    BalancePerProduct = data.Balance / (data.NumOfProducts + 1)
    # Prepare input for ONNX
    input_array = np.array([[data.Age, data.NumOfProducts, data.Balance, BalancePerProduct]], dtype=np.float32)
    
    # Run inference
    output = ort_session.run(None, {"input": input_array})
    print(output)
    predicted_class = int(output[0][0])  # 0 or 1
    class_probabilities = output[1][0]   # dict-like probs for class 0 and 1

    churn_label = "Customer will churn" if predicted_class == 1 else "Customer will not churn"
    churn_probability = class_probabilities[1]  # probability of churn

    return {
        "prediction": churn_label,
        "churn_probability": f"{churn_probability * 100:.2f}%"  # formatted nicely
    } 