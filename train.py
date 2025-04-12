from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import (
    # Bart
    BartTokenizer,
    BartForConditionalGeneration,
    
    # Pegasus
    PegasusTokenizer,
    PegasusForConditionalGeneration,

    # T5
    T5Tokenizer,
    T5ForConditionalGeneration,

    # Mistral
    AutoTokenizer, 
    AutoModelForCausalLM,

    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
import numpy as np
import logging
import psutil
import humanize
import torch
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_interval=50):
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            self._log_memory()

    def _log_memory(self):
        mem = psutil.virtual_memory()
        logger.info(
            f"\nCPU Memory - Used: {humanize.naturalsize(mem.used)} | "
            f"Free: {humanize.naturalsize(mem.available)}"
        )

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                free = total - mem
                logger.info(
                    f"GPU {i} Memory - Used: {humanize.naturalsize(mem)} | "
                    f"Free: {humanize.naturalsize(free)} | "
                    f"Total: {humanize.naturalsize(total)}"
                )


class TrainingSummarizationPipeline:
    MODEL_MAP = {
        'bart': {
            'tokenizer': BartTokenizer,
            'model': BartForConditionalGeneration,
            'base_name':'facebook/bart-large-cnn',
            'prefix': 'summarize: ',
            'type': 'encoder_decoder'
        },
        'pegasus': {
            'tokenizer': PegasusTokenizer,
            'model': PegasusForConditionalGeneration,
            'base_name':'google/pegasus-large',
            'prefix': '',
            'type': 'encoder_decoder'
        },
        't5': {
            'tokenizer': T5Tokenizer,
            'model': T5ForConditionalGeneration,
            'base_name': 'google/flan-t5-xl',
            'prefix': 'summarize: ',
            'type': 'encoder_decoder'
        },
        'trelis': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'base_name': 'Trelis/Mistral-7B-Instruct-v0.1-Summarize-64k',
            'prefix': '[INST] Summarize the following academic paper:\n\n',
            'type': 'causal'
        },
        'mistral instruct': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'base_name':'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
            'prefix': '[INST] Summarize the following scientific article.\n\n',
            'suffix': '[/INST]',
            'type': 'causal'
        },
        'mistral base': {
            'tokenizer': AutoTokenizer,
            'model': AutoModelForCausalLM,
            'base_name':'mistralai/Mistral-Small-3.1-24B-Base-2503',
            'prefix': 'Summarize the following scientific article.\n\n',
            'type': 'causal'
        }
    }

    def __init__(self, config_path='./config.json'):
        """Initialize the pipeline with configuration"""
        self.eval_dataset = None
        self.trainer = None
        self.train_dataset = None
        self.tokenized_dataset = None
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.config_path = config_path
        self.config = self._load_config()

        # Set device based on config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config['train']['cpu'] else "cpu")
        logger.info(f"Using device: {self.device}")

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {self.config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def _get_data_directory(self):
        """Generate data directory path from config"""
        data_config = self.config['data']
        dir_name = f"{int(data_config['train'] * 100)}-{int(data_config['test'] * 100)}-{int(data_config['eval'] * 100)}-{data_config['total']}"
        return f"./Data/Treated/{dir_name}"

    def load_data(self):
        """Load dataset from JSON files based on config"""
        data_dir = self._get_data_directory()
        train_path = os.path.join(data_dir, "train.json")
        test_path = os.path.join(data_dir, "test.json")

        logger.info(f"Loading dataset from {data_dir}...")
        self.dataset = load_dataset(
            'json',
            data_files={'train': train_path, 'test': test_path},
            streaming=True
        )
        return self.dataset

    def initialize_model(self):
        """Load tokenizer and model based on config"""
        model_name = self.config['model']['name']
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_MAP.keys())}")
        
        model_info = self.MODEL_MAP[model_name]
        logger.info(f"Initializing {model_name} model with base {model_info['base_name']}...")
        
        self.tokenizer = model_info['tokenizer'].from_pretrained(model_info['base_name'])
        self.model = model_info['model'].from_pretrained(model_info['base_name']).to(self.device)
        
        # Set padding token if not already set (for Mistral models)
        if model_info['type'] == 'causal' and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        return self.tokenizer, self.model

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
            input_lengths = [len(self.tokenizer(f"{model_info['prefix']}{text}").input_ids) for text in examples['input_text']]
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

    def prepare_datasets(self):
        """Apply preprocessing and tokenization to datasets"""
        logger.info("Preprocessing datasets...")
        processed_dataset = self.dataset.map(
            self.preprocess,
            batched=True,
            remove_columns=['abstract', 'text', 'paper_id', 'title', 'section_names', 'domain']
        )

        self.tokenized_dataset = processed_dataset.map(
            self.tokenize_data,
            batched=True
        )

        self.train_dataset = self.tokenized_dataset["train"].shuffle(seed=42)
        self.eval_dataset = self.tokenized_dataset["test"].shuffle(seed=42)

        return self.train_dataset, self.eval_dataset

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


    def train(self):
        """Train the model with configured parameters"""
        logger.info("Setting up training...")
        train_config = self.config['train']
        model_info = self.MODEL_MAP[self.config['model']['name']]

        batch_size = 4  # per_device_train_batch_size
        gradient_accumulation_steps = 1
        samples_per_epoch = self.config['data']['total'] * self.config['data']['train']
        steps_per_epoch = samples_per_epoch // (batch_size * gradient_accumulation_steps)
        max_steps = int(steps_per_epoch * train_config['epochs'])

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=train_config['LR'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=train_config['epochs'],
            max_steps=max_steps,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='eval-rougelL',
            greater_is_better=True,
            logging_dir="./logs",
            logging_steps=50,
            fp16=torch.cuda.is_available() and not train_config['cpu'],
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),
                MemoryMonitorCallback(log_interval=50)
            ],
            data_collator=data_collator,
        )

        logger.info(f"Starting training for {max_steps} total steps...")
        training_results = self.trainer.train()
        logger.info("Training completed.")

        return training_results

    def evaluate(self):
        """Evaluate the trained model"""
        if not hasattr(self, 'trainer'):
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model...")
        eval_results = self.trainer.evaluate()
        logger.info("Evaluation completed.")

        return eval_results

    def save_model(self, output_dir='./Models'):
        if not hasattr(self, 'trainer'):
            raise ValueError("Model must be trained before saving")

        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")


def main():
    """Main execution function"""
    try:
        # Initialize pipeline with config
        pipeline = TrainingSummarizationPipeline(config_path='./config.json')

        # Load data
        pipeline.load_data()

        # Initialize model components
        pipeline.initialize_model()

        # Prepare datasets
        pipeline.prepare_datasets()

        # Train model
        training_results = pipeline.train()

        # Evaluate model
        eval_results = pipeline.evaluate()

        # Save model
        pipeline.save_model()

        logger.info("\nTraining Results:")
        logger.info(training_results)

        logger.info("\nEvaluation Results:")
        logger.info(eval_results)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()