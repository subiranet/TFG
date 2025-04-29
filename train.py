from base_pipeline import BaseSummarizationPipeline
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset
import psutil
import humanize
import logging
import os
import torch
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logging goes to stdout for proper line updates
)
logger = logging.getLogger(__name__)


class ClearMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()  # Clears unused GPU memory


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.last_message_length = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            self._log_memory(state)

    def _log_memory(self, state):
        # Clear previous message
        sys.stdout.write('\r' + ' ' * self.last_message_length + '\r')
        sys.stdout.flush()

        # Get memory info
        mem = psutil.virtual_memory()
        cpu_mem = f"CPU Memory - Used: {humanize.naturalsize(mem.used)} | Free: {humanize.naturalsize(mem.available)}"

        gpu_mem = ""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                free = total - mem
                gpu_mem += f" | GPU {i} - Used: {humanize.naturalsize(mem)} Free: {humanize.naturalsize(free)}"

        # Get training progress info
        loss_value = "N/A"
        if state.log_history and len(state.log_history) > 0:
            loss_value = f"{state.log_history[-1].get('loss', 'N/A'):.4f}"

        progress = (f"Step {state.global_step}/{state.max_steps} "
                    f"({state.global_step / state.max_steps:.1%}) | "
                    f"Loss: {loss_value}")

        # Combine all info
        message = f"{progress} | {cpu_mem}{gpu_mem}"

        # Print the message and store its length for next clear
        sys.stdout.write(message)
        sys.stdout.flush()
        self.last_message_length = len(message)


class TrainingSummarization(BaseSummarizationPipeline):
    def __init__(self, config_path='./config.json'):
        super().__init__(config_path)
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_data(self):
        """Load dataset from JSON files based on config"""
        data_dir = self._get_data_directory()
        train_path = os.path.join(data_dir, "train.json")
        test_path = os.path.join(data_dir, "test.json")

        logging.info(f"Loading dataset from {data_dir}...")
        self.dataset = load_dataset(
            'json',
            data_files={'train': train_path, 'test': test_path},
            streaming=True
        )
        return self.dataset

    def prepare_datasets(self):
        """Apply preprocessing and tokenization to datasets"""
        logging.info("Preprocessing datasets...")
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

    def train(self):
        """Train the model with configured parameters"""
        logging.info("Setting up training...")
        train_config = self.config['train']


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
            metric_for_best_model='eval-final_score',
            greater_is_better=True,
            logging_dir="./logs",
            logging_steps=10,
            fp16=torch.cuda.is_available() and not train_config['cpu'],
            report_to=[],  # Disable other logging to prevent interference
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Enable model parallelism if needed
        if hasattr(self.model, "parallelize"):
            self.model.parallelize()


        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics_model,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),
                MemoryMonitorCallback(log_interval=10),
                ClearMemoryCallback()
            ],
            data_collator=data_collator
        )

        # Clear CUDA cache before training
        torch.cuda.empty_cache()

        # Add a custom callback to monitor gradients
        class GradientMonitorCallback(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step % 10 == 0:  # Check every 10 steps
                    # Check if gradients are being computed
                    has_gradients = False
                    max_grad = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 0:
                                has_gradients = True
                                max_grad = max(max_grad, grad_norm)

                    logging.info(f"Step {state.global_step}: Has gradients: {has_gradients}, Max gradient: {max_grad}")

        # Add the gradient monitor callback
        self.trainer.add_callback(GradientMonitorCallback())

        # Ensure model is in training mode
        self.model.train()

        # Log model trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model has {trainable_params:,} trainable parameters")

        logging.info(f"Starting training for {500} total steps...")
        try:
            training_results = self.trainer.train()
            print()
            logging.info("Training completed.")
        except KeyboardInterrupt:
            print()
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error("CUDA Out of Memory error occurred. Try further reducing batch sizes.")
                raise RuntimeError("Consider reducing batch sizes further or using a smaller model.") from e
            raise

        return training_results

    def save_model(self, output_dir='./Models'):
        if not hasattr(self, 'trainer'):
            raise ValueError("Model must be trained before saving")

        output_dir = f'{output_dir}/{self.config['model']['name']}-{self.dir_name}'
        os.mkdir(output_dir)

        logging.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logging.info("Model saved successfully")


def main():
    try:
        # Initialize trainer
        trainer = TrainingSummarization()

        # Load data
        trainer.load_data()

        # Initialize model components
        trainer.initialize_model()

        # Prepare datasets
        trainer.prepare_datasets()

        # Train
        results = trainer.train()

        # Save model
        trainer.save_model()

        logger.info(f'\nTraining results for model {trainer.config['model']['name']}:')
        logger.info(results)


    except Exception as e:
        print(f"Training execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
