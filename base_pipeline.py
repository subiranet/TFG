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
from fast_bleu import BLEU
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
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.length_penalty_alpha = 1.0

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
        """Compute BLEU and ROUGE (1, 2, L) metrics with length penalty"""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Tokenize for BLEU
        references = [[ref.split()] for ref in decoded_labels]
        candidates = [pred.split() for pred in decoded_preds]

        # Compute BLEU using FastBLEU
        bleu_scorer = BLEU(references)
        bleu_scores = bleu_scorer.get_score(candidates)
        bleu_score = np.mean(list(bleu_scores.values()))

        # Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        rouge1_list, rouge2_list, rougeL_list = [], [], []
        for ref, pred in zip(decoded_labels, decoded_preds):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_list.append(scores['rouge1'].fmeasure)
            rouge2_list.append(scores['rouge2'].fmeasure)
            rougeL_list.append(scores['rougeL'].fmeasure)

        rouge1_f1 = np.mean(rouge1_list)
        rouge2_f1 = np.mean(rouge2_list)
        rougeL_f1 = np.mean(rougeL_list)

        # Compute length ratio and penalty
        pred_lengths = [len(pred.split()) for pred in decoded_preds]
        label_lengths = [len(label.split()) for label in decoded_labels]
        length_ratio = np.mean([p / l if l != 0 else 0 for p, l in zip(pred_lengths, label_lengths)])
        length_penalty = min(1.0, length_ratio ** self.length_penalty_alpha)

        # F1-style combo of BLEU and ROUGE-L
        if bleu_score + rougeL_f1 > 0:
            f1_combo = 2 * bleu_score * rougeL_f1 / (bleu_score + rougeL_f1)
        else:
            f1_combo = 0.0

        final_score = f1_combo * length_penalty

        return {
            'bleu': bleu_score,
            'rouge1': rouge1_f1,
            'rouge2': rouge2_f1,
            'rougeL': rougeL_f1,
            'length_ratio': length_ratio,
            'combined_f1': f1_combo,
            'final_score': final_score
        }
