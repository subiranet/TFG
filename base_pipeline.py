from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)
from rouge_score import rouge_scorer
import logging
import json
import torch
import numpy as np



class BaseSummarizationPipeline:
    MODEL_MAP = {
        'bart': {
            'tokenizer': BartTokenizer,
            'model': BartForConditionalGeneration,
            'prefix': 'summarize: ',
            'type': 'encoder_decoder'
        },
        'pegasus': {
            'tokenizer': PegasusTokenizer,
            'model': PegasusForConditionalGeneration,
            'prefix': '',
            'type': 'encoder_decoder'
        },
        't5': {
            'tokenizer': T5Tokenizer,
            'model': T5ForConditionalGeneration,
            'prefix': 'summarize: ',
            'type': 'encoder_decoder'
        },
        'trelis': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'prefix': '[INST] Summarize the following academic paper:\n\n',
            'type': 'causal'
        },
        'mistral instruct': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'prefix': '[INST] Summarize the following scientific article.\n\n',
            'suffix': '[/INST]',
            'type': 'causal'
        },
        'mistral base': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'prefix': 'Summarize the following scientific article.\n\n',
            'type': 'causal'
        }
    }

    def __init__(self, config_path='./config.json'):
        """Initialize the base pipeline with common configuration"""
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config['train']['cpu'] else "cpu")

        logging.info(f"Using device: {self.device}")

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Config file {self.config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing config file: {e}")
            raise

    def _get_data_directory(self):
        """Generate data directory path from config"""
        data_config = self.config['data']
        dir_name = (f"{int(data_config['train'] * 100)}-"
                    f"{int(data_config['test'] * 100)}-"
                    f"{int(data_config['eval'] * 100)}-"
                    f"{data_config['total']}")

        return f"./Data/Treated/{dir_name}"

    @staticmethod
    def merge_sections(text, section_names):
        """Merge document sections with their titles"""
        return '\n\n'.join([f'{title}: {"".join(content)}' for title, content in zip(section_names, text)])

    def preprocess(self, examples):
        """Process individual examples before tokenization"""
        processed = {'input_text': [], 'target_text': []}

        for i in range(len(examples['title'])):
            merged_text = self.merge_sections(examples['text'][i], examples['section_names'][i])
            input_text = (
                f"Title: {examples['title'][i]}\n"
                f"Domains: {', '.join(examples['domain'][i])}\n\n"
                f"{merged_text}"
            )
            target_text = ''.join(examples['abstract'][i])

            processed['input_text'].append(input_text)
            processed['target_text'].append(''.join(target_text))

        return processed

    def tokenize_data(self, examples):
        """Tokenize the processed examples with special handling for Mistral models"""
        model_info = self.MODEL_MAP[self.config['model']['name']]

        if model_info['type'] == 'encoder_decoder':
            # Standard encoder-decoder models (BART, Pegasus, T5)
            inputs = [f"{model_info['prefix']}{text}" for text in examples['input_text']]
            model_inputs = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            labels = self.tokenizer(
                text_target=examples['target_text'],
                padding='max_length',
                truncation=True,
                max_length=150
            )

            model_inputs["labels"] = labels["input_ids"]

        else:
            # Causal models (Mistral and variants)
            full_texts = []
            for input_text, target_text in zip(examples['input_text'], examples['target_text']):
                # Format the text according to the model's instruction format
                instruction = f"{model_info['prefix']}{input_text}"
                if 'suffix' in model_info:
                    instruction += f" {model_info['suffix']}"
                full_text = f"{instruction} {target_text}"
                full_texts.append(full_text)

            # Tokenize the full sequence
            tokenized = self.tokenizer(
                full_texts,
                max_length=512 + 150,  # Input + target length
                truncation=True,
                padding='max_length'
            )

            # Create labels by masking the input part
            input_lengths = [len(self.tokenizer(f"{model_info['prefix']}{text}").input_ids) for text in
                             examples['input_text']]
            labels = []
            for i, length in enumerate(input_lengths):
                label = [-100] * length  # Mask the input part
                label += tokenized['input_ids'][i][length:]  # Keep the target part
                labels.append(label)

            model_inputs = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            }

        return model_inputs

    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation"""
        preds, labels = eval_pred

        # Handle tuple output (e.g., from seq2seq models)
        if isinstance(preds, tuple):
            preds = preds[0]  # Take the first element (logits)

        # Convert logits to token IDs (if needed)
        if preds.ndim == 3:  # Shape: [batch_size, seq_len, vocab_size]
            preds = np.argmax(preds, axis=-1)

        # Ensure we have integer token IDs
        preds = np.array(preds, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)

        # Replace -100 (masked tokens) with pad_token_id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        rouge_scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }

        for pred, label in zip(decoded_preds, decoded_labels):
            scores = self.scorer.score(pred, label)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

        return {
            "rouge-1": np.mean(rouge_scores["rouge1"]),
            "rouge-2": np.mean(rouge_scores["rouge2"]),
            "rouge-L": np.mean(rouge_scores["rougeL"]),
        }
