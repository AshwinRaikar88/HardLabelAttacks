import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import os

class MistralClassifier:
    """Mistral classifier using logits-based prediction"""

    def __init__(self, model_path, dataset="rotten_tomatoes",hf_token=None, device="cuda"):
        self.device = device
        self.dataset = dataset

        if self.dataset == 'imdb':
            self.max_seq_length = 2048
        else:
            self.max_seq_length = 512

        # Dataset-specific configurations
        self.dataset_configs = {
            'rotten_tomatoes': {
                'labels': ['negative', 'positive'],
                'num_classes': 2,
                'task_type': 'sentiment'
            },
            'imdb': {
                'labels': ['negative', 'positive'],
                'num_classes': 2,
                'task_type': 'sentiment'
            },
            'ag_news': {
                'labels': ['world', 'sports', 'business', 'sci/tech'],
                'num_classes': 4,
                'task_type': 'topic'
            }
        }

        if dataset not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset}")

        config = self.dataset_configs[dataset]
        self.labels = config['labels']
        self.num_classes = config['num_classes']
        self.task_type = config['task_type']

        # Label mapping (lowercase for token matching)
        self.label_map = {i: label.lower() for i, label in enumerate(self.labels)}
        self.reverse_label_map = {label.lower(): i for i, label in enumerate(self.labels)}

        # Load model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_adapter = os.path.exists(adapter_config_path)

        if is_adapter:
            print(f"Loading base model: unsloth/mistral-7b-instruct-v0.3")
            print(f"Loading LoRA adapters from: {model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/mistral-7b-instruct-v0.3",
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
                token=hf_token,
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print(f"Loading full model from: {model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
                token=hf_token,
            )

        FastLanguageModel.for_inference(self.model)
        print(f"Mistral classifier loaded for {dataset} dataset!\n")

    def _format_prompt(self, text):
        """Format prompt based on dataset type"""
        if self.task_type == 'sentiment':
            prompt = f"Label the following input as one of the following labels: Positive, Negative.\n\nReview: {text}"
        else:  # topic classification
            labels_str = ", ".join([l.capitalize() for l in self.labels])
            prompt = f"Label the following text as one of the following categories: {labels_str}.\n\nText: {text}"

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def predict_logits(self, text):
        """Predict using logits"""
        formatted_prompt = self._format_prompt(text)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get logits for the last token position
        score_start = logits[:, -1, :]
        scores = {}

        # Extract scores for each label
        for label in self.label_map.values():
            cur_id = self.tokenizer.convert_tokens_to_ids(label)

            if cur_id == self.tokenizer.unk_token_id or cur_id is None:
                encoded = self.tokenizer.encode(label, add_special_tokens=False)
                cur_id = encoded[0] if len(encoded) > 0 else self.tokenizer.unk_token_id

            scores[label] = score_start[0, cur_id].item()

        # Find label with highest score
        pred_label_str = max(scores.keys(), key=lambda x: scores[x])
        pred_label_idx = self.reverse_label_map[pred_label_str]
        confidence = scores[pred_label_str]

        return pred_label_idx, confidence, scores