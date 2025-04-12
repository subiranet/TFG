from base_pipeline import BaseSummarizationPipeline
from transformers import Trainer, DataCollatorWithPadding
from datasets import load_dataset
import logging
import os



class EvaluationSummarizationPipeline(BaseSummarizationPipeline):
    def __init__(self, config_path='./config.json', model_dir='./Models'):
        super().__init__(config_path)
        self.eval_dataset = None
        self.model_dir = model_dir

    def load_data(self):
        """Load evaluation dataset from JSON files based on config"""
        data_dir = self._get_data_directory()
        test_path = os.path.join(data_dir, "test.json")

        logging.info(f"Loading evaluation dataset from {data_dir}...")
        self.dataset = load_dataset(
            'json',
            data_files={'test': test_path},
            streaming=True
        )
        return self.dataset

    def load_model(self):
        """Load trained model and tokenizer"""
        model_name = self.config['model']['name']
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_MAP.keys())}")

        model_info = self.MODEL_MAP[model_name]
        logging.info(f"Loading {model_name} model from {self.model_dir}...")

        self.tokenizer = model_info['tokenizer'].from_pretrained(self.model_dir)
        self.model = model_info['model'].from_pretrained(self.model_dir).to(self.device)

        # Set padding token if not already set (for Mistral models)
        if model_info['type'] == 'causal' and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        return self.tokenizer, self.model

    def prepare_dataset(self):
        """Apply preprocessing and tokenization to evaluation dataset"""
        logging.info("Preprocessing evaluation dataset...")
        processed_dataset = self.dataset.map(
            self.preprocess,
            batched=True,
            remove_columns=['abstract', 'text', 'paper_id', 'title', 'section_names', 'domain']
        )

        self.tokenized_dataset = processed_dataset.map(
            self.tokenize_data,
            batched=True
        )

        self.eval_dataset = self.tokenized_dataset["test"].shuffle(seed=42)
        return self.eval_dataset

    def evaluate(self):
        """Evaluate the trained model"""
        logging.info("Setting up evaluation...")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )

        logging.info("Starting evaluation...")
        eval_results = trainer.evaluate()
        logging.info("Evaluation completed.")

        return eval_results
