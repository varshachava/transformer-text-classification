from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "sentiment-analysis",
    model="bert-base-uncased"
)

@app.get("/predict")
def predict(text: str):
    result = classifier(text)
    return {"prediction": result}
