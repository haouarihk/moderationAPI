import json
import torch
import uuid
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Dict, List, Optional

# Initialize FastAPI app
app = FastAPI()  # This line must be here

# Model configuration
DEFAULT_MODEL_NAME = "ifmain/ModerationBERT-En-02"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model cache
model_cache = {}
current_model_name = DEFAULT_MODEL_NAME

def load_model(model_name: str) -> None:
    global current_model_name
    if model_name in model_cache:
        # Model already loaded, just update current model name
        current_model_name = model_name
        return
    
    try:
        if "bert" not in model_name.lower():
            return
        
        # Load new model and add to cache
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=18)
        model.to(device)
        model_cache[model_name] = (tokenizer, model)
        current_model_name = model_name
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load model {model_name}. Please ensure it's a valid BERT model. Error: {str(e)}"
        )

# Load default model at startup
load_model(DEFAULT_MODEL_NAME)

def get_current_model():
    return model_cache[current_model_name]

# Define categories and their OpenAI mappings
category_mappings = {
    'harassment': 'harassment',
    'harassment_threatening': 'harassment/threatening',
    'hate': 'hate',
    'hate_threatening': 'hate/threatening',
    'self_harm': 'self-harm',
    'self_harm_instructions': 'self-harm/instructions',
    'self_harm_intent': 'self-harm/intent',
    'sexual': 'sexual',
    'sexual_minors': 'sexual/minors',
    'violence': 'violence',
    'violence_graphic': 'violence/graphic'
}

# Define a threshold for flagging
threshold = 0.5

class ModerationInput(BaseModel):
    input: str | List[str]
    model: str | None = None

class Categories(BaseModel):
    sexual: bool
    hate: bool
    harassment: bool
    self_harm: bool
    sexual_minors: bool
    hate_threatening: bool
    violence_graphic: bool
    self_harm_intent: bool
    self_harm_instructions: bool
    harassment_threatening: bool
    violence: bool

class CategoryScores(BaseModel):
    sexual: float
    hate: float
    harassment: float
    self_harm: float
    sexual_minors: float
    hate_threatening: float
    violence_graphic: float
    self_harm_intent: float
    self_harm_instructions: float
    harassment_threatening: float
    violence: float

class ModerationResult(BaseModel):
    flagged: bool
    categories: Categories
    category_scores: CategoryScores

class ModerationResponse(BaseModel):
    id: str
    model: str
    results: List[ModerationResult]

def predict(text: str):
    tokenizer, model = get_current_model()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.sigmoid(outputs.logits)  # Convert logits to probabilities
    return predictions

def process_predictions(predictions):
    # Convert predictions to a dictionary with OpenAI category names
    raw_scores = {category_mappings[cat]: predictions[0][i].item() 
                 for i, cat in enumerate(category_mappings.keys())}
    
    # Create categories object (boolean flags)
    categories = Categories(
        sexual=raw_scores['sexual'] > threshold,
        hate=raw_scores['hate'] > threshold,
        harassment=raw_scores['harassment'] > threshold,
        self_harm=raw_scores['self-harm'] > threshold,
        sexual_minors=raw_scores['sexual/minors'] > threshold,
        hate_threatening=raw_scores['hate/threatening'] > threshold,
        violence_graphic=raw_scores['violence/graphic'] > threshold,
        self_harm_intent=raw_scores['self-harm/intent'] > threshold,
        self_harm_instructions=raw_scores['self-harm/instructions'] > threshold,
        harassment_threatening=raw_scores['harassment/threatening'] > threshold,
        violence=raw_scores['violence'] > threshold
    )
    
    # Create category_scores object
    category_scores = CategoryScores(
        sexual=raw_scores['sexual'],
        hate=raw_scores['hate'],
        harassment=raw_scores['harassment'],
        self_harm=raw_scores['self-harm'],
        sexual_minors=raw_scores['sexual/minors'],
        hate_threatening=raw_scores['hate/threatening'],
        violence_graphic=raw_scores['violence/graphic'],
        self_harm_intent=raw_scores['self-harm/intent'],
        self_harm_instructions=raw_scores['self-harm/instructions'],
        harassment_threatening=raw_scores['harassment/threatening'],
        violence=raw_scores['violence']
    )
    
    # Determine if any category is flagged
    flagged = any(getattr(categories, field) for field in categories.__fields__)
    
    return ModerationResult(
        flagged=flagged,
        categories=categories,
        category_scores=category_scores
    )

@app.post("/v1/moderations", response_model=ModerationResponse)
async def moderate_text(request: ModerationInput):
    # Generate a unique ID for the request
    request_id = f"modr-{str(uuid.uuid4())[:8]}"
    
    # Load custom model if provided and different from current
    if request.model and request.model != current_model_name:
        load_model(request.model)
    
    # Convert single string input to list for consistent processing
    input_texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Process each input text
    results = []
    for text in input_texts:
        predictions = predict(text)
        result = process_predictions(predictions)
        results.append(result)
    
    return ModerationResponse(
        id=request_id,
        model=current_model_name,  # Return the actual model name being used
        results=results
    )
