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
    """
    Callback to clear GPU memory after each training step.

    Helps prevent memory accumulation during long training runs.
    """
    def on_step_end(self, args, state, control, **kwargs):
        """Clear CUDA cache after each step to free unused memory"""
        torch.cuda.empty_cache()


class MemoryMonitorCallback(TrainerCallback):
    """
    Callback to monitor and log memory usage during training.

    Tracks both CPU and GPU memory usage and reports at regular intervals
    and at the end of each epoch.
    """
    def __init__(self, log_interval=50):
        """
        Initialize the memory monitor.

        Args:
            log_interval: Number of steps between memory usage logs
        """
        self.log_interval = log_interval
        self.last_epoch = -1

    def on_step_end(self, args, state, control, **kwargs):
        """
        Log memory usage at regular step intervals.

        Args:
            args: Training arguments
            state: Current training state
            control: Trainer control object
            **kwargs: Additional arguments
        """
        if state.global_step % self.log_interval == 0:
            self._log_memory(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Log completion message at the end of each epoch.

        Args:
            args: Training arguments
            state: Current training state
            control: Trainer control object
            **kwargs: Additional arguments
        """
        current_epoch = state.epoch
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            logger.info(f"Completed epoch {int(current_epoch)}/{args.num_train_epochs} ({current_epoch/args.num_train_epochs:.1%})")

    def _log_memory(self, state):
        """
        Collect and log memory usage information.

        Gathers CPU and GPU memory statistics and combines with
        training progress information for comprehensive logging.

        Args:
            state: Current training state containing step and loss information
        """
        # Get CPU memory info
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

        # Calculate current epoch (may be fractional)
        current_epoch = state.epoch

        # Combine all info
        progress = (f"Step {state.global_step}/{state.max_steps} "
                   f"({state.global_step / state.max_steps:.1%}) | "
                   f"Epoch {current_epoch:.2f}/{state.num_train_epochs} | "
                   f"Loss: {loss_value}")

        # Log the message using the logger
        logger.info(f"{progress} | {cpu_mem}{gpu_mem}")


class TrainingSummarization(BaseSummarizationPipeline):
    def __init__(self, config_path='./config.json'):
        super().__init__(config_path)
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_data(self):
        """
        Load dataset from JSON files based on configuration.

        Uses the data directory path from _get_data_directory() to locate
        the train.json and test.json files, then loads them as a dataset.

        Returns:
            The loaded dataset object
        """
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

    def prepare_datasets(self):
        """
        Apply preprocessing and tokenization to datasets.

        Processes the raw dataset by:
        1. Applying the preprocess method to format inputs and targets
        2. Tokenizing the processed data for model input
        3. Creating separate train and evaluation datasets

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
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

    def train(self):
        """
        Train the model with configured parameters.

        Sets up training arguments, initializes the trainer with appropriate
        callbacks for monitoring, and runs the training process.

        Returns:
            Training results from the Hugging Face Trainer

        Raises:
            RuntimeError: If CUDA out of memory error occurs
            KeyboardInterrupt: If training is interrupted by user
        """
        # Configure CUDA memory allocation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("Setting up training...")
        train_config = self.config['train']


        batch_size = 4  # per_device_train_batch_size
        desired_effective_batch_size = 1000
        gradient_accumulation_steps = desired_effective_batch_size // batch_size

        samples_per_epoch = self.config['data']['total'] * self.config['data']['train']
        steps_per_epoch = samples_per_epoch // (batch_size * gradient_accumulation_steps)
        max_steps = int(steps_per_epoch * train_config['epochs'])

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            learning_rate=train_config['LR'],
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=train_config['epochs'],
            max_steps=max_steps,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='eval-final_score',
            gradient_checkpointing=True,
            greater_is_better=True,
            logging_dir="./logs",
            logging_steps=5,
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
                MemoryMonitorCallback(log_interval=5),
                ClearMemoryCallback()
            ],
            data_collator=data_collator
        )

        # Clear CUDA cache before training
        torch.cuda.empty_cache()

        # Add a custom callback to monitor gradients
        class GradientMonitorCallback(TrainerCallback):
            """
            Callback to monitor gradient flow during training.

            Checks if gradients are being properly computed and reports
            the maximum gradient norm to help diagnose training issues.
            """
            def __init__(self, log_interval=10):
                """
                Initialize the gradient monitor.

                Args:
                    log_interval: Number of steps between gradient checks
                """
                self.log_interval = log_interval

            def on_step_end(self, args, state, control, model=None, **kwargs):
                """
                Check and log gradient information at regular intervals.

                Args:
                    args: Training arguments
                    state: Current training state
                    control: Trainer control object
                    model: The model being trained
                    **kwargs: Additional arguments
                """
                if state.global_step % self.log_interval == 0:
                    # Check if gradients are being computed
                    has_gradients = False
                    max_grad = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 0:
                                has_gradients = True
                                max_grad = max(max_grad, grad_norm)

                    logger.info(f"Gradient info - Step {state.global_step}: Has gradients: {has_gradients}, Max gradient: {max_grad:.6f}")

        # Add the gradient monitor callback
        self.trainer.add_callback(GradientMonitorCallback(log_interval=20))

        # Ensure model is in training mode
        self.model.train()

        # Log model trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model has {trainable_params:,} trainable parameters")

        logger.info(f"Starting training for {max_steps} total steps...")
        try:
            training_results = self.trainer.train()
            logger.info("Training completed.")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory error occurred. Try further reducing batch sizes.")
                raise RuntimeError("Consider reducing batch sizes further or using a smaller model.") from e
            raise

        return training_results

    def save_model(self, output_dir='./Models'):
        """
        Save the trained model and tokenizer to disk.

        Creates a directory with the model name and data configuration,
        then saves both the model and tokenizer to that location.

        Args:
            output_dir: Base directory where the model will be saved

        Raises:
            ValueError: If the model hasn't been trained yet
        """
        if not hasattr(self, 'trainer'):
            raise ValueError("Model must be trained before saving")

        output_dir = f'{output_dir}/{self.config['model']['name']}-{self.dir_name}'
        os.mkdir(output_dir)

        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")


def main():
    """
    Main entry point for the training script.

    Orchestrates the complete training pipeline:
    1. Initializes the training pipeline
    2. Loads and prepares datasets
    3. Initializes the model
    4. Trains the model
    5. Saves the trained model

    Handles exceptions and provides error reporting.
    """
    try:
        # Initialize training pipeline
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
