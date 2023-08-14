from transformers import pipeline
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class SentenceRequest(BaseModel):
    sentence: str

class SentenceResponse(BaseModel):
    labels: dict

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
@app.post("/sentimentclassifier", response_model=SentenceResponse)
async def classify_sentence(request: SentenceRequest = Body(...)):
    sentence = request.sentence
    candidate_labels = [
    'Anger',
    'Confidence/Inspiration/Proud/Hopeful',
    'Confusion',
    'Disgust',
    'Envy',
    'Excitement',
    'Exhausted',
    'Fear',
    'Gratitude',
    'Grief',
    'Guilt',
    'Happiness/Joy',
    'Helplessness',
    'Hurt',
    'Loneliness',
    'Love',
    'Neutral Emotion',
    'Numbness',
    'Regret',
    'Relieved',
    'Sad',
    'Satisfaction/Contentment/Peace',
    'Shame',
    'Surprise',
    'Worthlessness'
    ]
    output = classifier(sentence, candidate_labels, multi_label=True)
    labels = {}
    for i in range(len(output.get('scores'))):
        if output.get("scores")[i] > 0.9:
            labels[output.get("labels")[i]] = output.get('scores')[i]
    if len(labels)==0:
        labels[output.get('labels')[0]] = output.get('scores')[0]
    response = SentenceResponse(labels=labels)
    return response

# uvicorn [filename]:app --reload