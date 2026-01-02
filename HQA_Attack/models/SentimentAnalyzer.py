import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import pipeline


# class SentimentAnalyzer:
#     def __init__(self):
#         # Load a pre-trained sentiment analysis pipeline
#         self.model = pipeline("sentiment-analysis",
#                                     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        
#     def get_preds(self, text):
#         return self.model(text)
    
#     def get_pred_labels(self, text):        
#         return self.get_preds(text)[0]['label']

class SentimentAnalyzer:
    def __init__(self, model_dict, data="imdb"):
        self.model = pipeline(model_dict[data][0], model=model_dict[data][1], truncation=True)

        self.dataset = load_dataset(model_dict[data][2], split=model_dict[data][3])

        self.dataset_sample = self.dataset.shuffle(seed=30).select(range(1000))

        print(f"model: {model_dict[data][1]}")        
        print(f"dataset: {model_dict[data][2]}")

    def get_preds(self, text):
        return self.model(text)

    def get_pred_labels(self, text):
        return self.get_preds(text)[0]['label']


class MultilingualBERT:
    def __init__(self):
        
        # Load a fine-tuned BERT model
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class = torch.argmax(logits).item()
        return predicted_class
