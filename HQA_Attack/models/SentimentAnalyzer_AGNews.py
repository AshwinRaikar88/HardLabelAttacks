from transformers import pipeline
from datasets import load_dataset

class SentimentAnalyzer_AG_News:
    def __init__(self):
        self.model = pipeline("text-classification", model="textattack/distilbert-base-uncased-ag-news")

        self.dataset = load_dataset("fancyzhx/ag_news", split="test")

        self.dataset_sample = self.dataset.shuffle(seed=30).select(range(1000))

        print("model: textattack/distilbert-base-uncased-ag-news")
        print("dataset: fancyzhx/ag_news")

    def get_preds(self, text):
        return self.model(text)

    def get_pred_labels(self, text):
        return self.get_preds(text)[0]['label']