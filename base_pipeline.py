from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
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
            'base_name': 'facebook/bart-large-cnn',
            'prefix': 'summarize: ',
            'type': 'encoder_decoder'
        },
        'pegasus': {
            'tokenizer': PegasusTokenizer,
            'model': PegasusForConditionalGeneration,
            'base_name': 'google/pegasus-large',
            'prefix': '',
            'type': 'encoder_decoder'
        },
        'arxiv': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForSeq2SeqLM,
            'base_name': 'google/bigbird-pegasus-large-arxiv',
            'prefix': '',
            'type': 'encoder_decoder'
        },
        'book': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForSeq2SeqLM,
            'base_name': 'pszemraj/led-large-book-summary',
            'prefix': 'Summarize the following scientific article and create the abstract.\n\n',
            'type': 'encoder_decoder'
        },
        'tinyLlama': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'base_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'prefix': 'Summarize the following scientific article and create the abstract.\n\n',
            'type': 'causal'
        },
        'scibert': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForSeq2SeqLM,
            'base_name': 'allenai/led-large-16384',
            'prefix': 'Summarize the following scientific article and create the abstract.\n\n',
            'type': 'encoder_decoder'
        }
    }

    def __init__(self, config_path='./config.json'):
        """
        Initialize the base summarization pipeline.

        Sets up the pipeline with configuration, device detection, and metric scoring.
        Does not load models or datasets by default - use initialize_model() for that.

        Args:
            config_path: Path to the JSON configuration file
        """
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config['train']['cpu'] else "cpu")
        self.length_penalty_alpha = 1.0

        logging.info(f"Using device: {self.device}")

    def _load_config(self):
        """
        Load configuration from JSON file.

        Reads and parses the configuration file specified in self.config_path.

        Returns:
            Dictionary containing configuration parameters

        Raises:
            FileNotFoundError: If the config file doesn't exist
            JSONDecodeError: If the config file contains invalid JSON
        """
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
        """
        Generate data directory path from configuration.

        Creates a directory name based on the data split percentages and total count.
        Format: "{train%}-{test%}-{eval%}-{total}"

        Returns:
            String path to the data directory
        """
        data_config = self.config['data']
        self.dir_name = (f"{int(data_config['train'] * 100)}-"
                         f"{int(data_config['test'] * 100)}-"
                         f"{int(data_config['eval'] * 100)}-"
                         f"{data_config['total']}")

        return f"./Data/Treated/{self.dir_name}"

    @staticmethod
    def merge_sections(text, section_names):
        """
        Merge document sections with their corresponding titles.

        Args:
            text: List of section content (each section can be a list of strings)
            section_names: List of section titles

        Returns:
            String with formatted sections where each section has its title
        """
        return '\n\n'.join([f'{title}: {"".join(content)}' for title, content in zip(section_names, text)])

    @staticmethod
    def preprocess(examples):
        """
        Process individual examples before tokenization.

        Formats the input by combining title, domains, and section text.
        Creates a structured input with clear section headers.

        Args:
            examples: Dictionary containing raw dataset examples with keys:
                     'title', 'text', 'section_names', 'domain', 'abstract'

        Returns:
            Dictionary with 'input_text' and 'target_text' lists
        """
        processed = {'input_text': [], 'target_text': []}

        for i in range(len(examples['title'])):
            merged_text = BaseSummarizationPipeline.merge_sections(examples['text'][i], examples['section_names'][i])
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
        """
        Tokenize the processed examples for model training or inference.

        Handles different tokenization approaches based on model type:
        - For encoder-decoder models (BART, Pegasus, T5): Tokenizes inputs and targets separately
        - For causal models (Mistral, TinyLlama): Combines input and target with special formatting

        Args:
            examples: Dictionary containing 'input_text' and 'target_text' lists

        Returns:
            Dictionary with tokenized inputs ready for the model
        """
        model_info = self.MODEL_MAP[self.config['model']['name']]

        if model_info['type'] == 'encoder_decoder':
            # Standard encoder-decoder models (BART, Pegasus, T5)
            inputs = [f"{model_info['prefix']}{text}" for text in examples['input_text']]

            # Get model's max context length or default to 2048 if not specified
            model_max_length = getattr(self.tokenizer, 'model_max_length', 2048)

            # Ensure we don't exceed the model's maximum context length
            input_max_length = min(1024, model_max_length)
            target_max_length = min(150, model_max_length)

            model_inputs = self.tokenizer(
                inputs,
                max_length=input_max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            labels = self.tokenizer(
                text_target=examples['target_text'],
                padding='max_length',
                truncation=True,
                max_length=target_max_length,
                return_tensors='pt'
            )

            # Replace padding token id with -100 in labels to ignore loss on padding tokens
            labels_tensor = labels["input_ids"].clone()
            labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels_tensor

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

            # Get model's max context length or default to 2048 if not specified
            model_max_length = getattr(self.tokenizer, 'model_max_length', 2048)

            # Ensure we don't exceed the model's maximum context length
            max_length = min(512 + 150, model_max_length)  # Input + target length

            # Tokenize the full sequence
            tokenized = self.tokenizer(
                full_texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'  # Ensure we get PyTorch tensors
            )

            # Create labels by masking the input part
            # Ensure we respect the model's maximum context length when calculating input lengths
            input_lengths = [len(self.tokenizer(
                f"{model_info['prefix']}{text}", 
                truncation=True, 
                max_length=min(512, model_max_length)
            ).input_ids) for text in examples['input_text']]
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

    def compute_metrics(self, decoded_labels: list[str], decoded_preds: list[str]):
        """
        Compute evaluation metrics for summarization quality.

        Args:
            decoded_labels: List of reference/ground truth summaries
            decoded_preds: List of model-generated summaries

        Returns:
            Dictionary containing various metrics:
            - bleu: BLEU score
            - rouge1/2/L: ROUGE F1 scores
            - length_ratio: Ratio of prediction length to reference length
            - combined_f1: Harmonic mean of BLEU and ROUGE-L
            - final_score: Combined score with length penalty
        """
        references = [[ref.split()] for ref in decoded_labels]
        bleu_ref = [ref.split() for ref in decoded_labels]
        candidates = [pred.split() for pred in decoded_preds]

        bleu_scorer = BLEU(bleu_ref)
        bleu_scores = bleu_scorer.get_score(candidates)
        bleu_score = np.mean(list(bleu_scores.values()))

        rouge1_list, rouge2_list, rougeL_list = [], [], []
        for ref, pred in zip(decoded_labels, decoded_preds):
            scores = self.scorer.score(ref, pred)
            rouge1_list.append(scores['rouge1'].fmeasure)
            rouge2_list.append(scores['rouge2'].fmeasure)
            rougeL_list.append(scores['rougeL'].fmeasure)

        rouge1_f1 = np.mean(rouge1_list)
        rouge2_f1 = np.mean(rouge2_list)
        rougeL_f1 = np.mean(rougeL_list)

        pred_lengths = [len(pred.split()) for pred in decoded_preds]
        label_lengths = [len(label.split()) for label in decoded_labels]
        length_ratio = np.mean([p / l if l != 0 else float('inf') for p, l in zip(pred_lengths, label_lengths)])
        length_penalty = min(1.0, length_ratio ** self.length_penalty_alpha)

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

    def compute_metrics_model(self, eval_pred):
        """
        Compute metrics for model evaluation compatible with Hugging Face's Trainer.

        Args:
            eval_pred: Tuple of (predictions, labels) from model evaluation
                       Can contain tensors or numpy arrays

        Returns:
            Dictionary of metrics from compute_metrics method
        """
        predictions, labels = eval_pred

        # Handle the case where predictions are tuples (from generative models)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Convert predictions to numpy array if they're tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 (masked tokens) with pad_token_id before decoding labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return self.compute_metrics(decoded_labels, decoded_preds)


    def initialize_model(self):
        """
        Initialize tokenizer and model based on configuration.

        Loads the appropriate pre-trained model and tokenizer based on the
        model name specified in the configuration. Sets up padding tokens
        for causal language models if needed.

        Returns:
            Tuple of (tokenizer, model)

        Raises:
            ValueError: If the specified model is not supported
        """
        model_name = self.config['model']['name']
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_MAP.keys())}")

        model_info = self.MODEL_MAP[model_name]
        logging.info(f"Initializing {model_name} model with base {model_info['base_name']}...")

        self.tokenizer = model_info['tokenizer'].from_pretrained(model_info['base_name'])
        self.model = model_info['model'].from_pretrained(model_info['base_name']).to_empty(device=self.device)

        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        # Set padding token if not already set (for Mistral models)
        if model_info['type'] == 'causal' and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        return self.tokenizer, self.model

    def load_model_from_dir(self, model_dir):
        """
        Load a pre-trained model and tokenizer from a directory.

        Args:
            model_dir: Path to the directory containing the saved model

        Raises:
            ValueError: If the model type is not supported
            Exception: If there's an error loading the model
        """
        try:
            # Extract model name from config
            model_name = self.config['model']['name']
            if model_name not in self.MODEL_MAP:
                raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_MAP.keys())}")

            model_info = self.MODEL_MAP[model_name]
            logging.info(f"Loading {model_name} model from {model_dir}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Load model based on model type
            if model_info['type'] == 'encoder_decoder':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            elif model_info['type'] == 'causal':
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")

            # Set padding token if not already set (for Mistral models)
            if model_info['type'] == 'causal' and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            self.model.to_empty(device=self.device)
        except Exception as e:
            logging.error(f'Error loading local model {model_dir}:\n{e}')

    def generate_output(self, input_text, min_length=10, max_length=150, num_beams=4, temperature=1.0, top_k=50, top_p=0.95):
        """
        Generate a summary for the given input text using the loaded model.

        Args:
            input_text: The text to summarize
            min_length: Minimum length of the generated summary
            max_length: Maximum length of the generated summary
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated summary text

        Raises:
            RuntimeError: If model or tokenizer is not initialized
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer not initialized")

        model_specs = self.MODEL_MAP[self.config['model']['name']]
        self.model.to_empty(device=self.device)

        if model_specs['type'] == 'encoder_decoder':
            processed_input = f"{model_specs['prefix']}{input_text}"
            max_context = getattr(self.tokenizer, 'model_max_length', 2048)

            inputs = self.tokenizer(
                processed_input,
                return_tensors="pt",
                truncation=True,
                max_length=min(512, max_context)
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                min_length=min_length,
                max_length=min(max_length, max_context),
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=3
            )
        else:
            processed_input = f"{model_specs['prefix']}{input_text}"
            if 'suffix' in model_specs:
                processed_input += f" {model_specs['suffix']}"

            max_context = getattr(self.tokenizer, 'model_max_length', 2048)
            inputs = self.tokenizer(
                processed_input,
                return_tensors="pt",
                truncation=True,
                max_length=min(512, max_context)
            ).to(self.device)

            input_length = inputs['input_ids'].shape[1]
            outputs = self.model.generate(
                **inputs,
                max_length=min(input_length + max_length, max_context),
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )[:, input_length:]

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
