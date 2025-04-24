import logging
import os
import sys
import json
import numpy as np

from datasets import load_dataset

from evaluate import EvaluationSummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logging goes to stdout for proper line updates
)
logger = logging.getLogger(__name__)


class TestModel(EvaluationSummarizationPipeline):
    def __init__(self, config_path='./config.json', model_dir='./Models'):
        super().__init__(config_path=config_path, model_dir=model_dir)
        self.generated_outputs = None
        self.data = None
        self.eval_results = None
        self.paper_id = None

        # Set paper_id based on config
        eval_config = self.config.get('eval', {})
        if eval_config.get('ids') and len(eval_config['ids']) > 0:
            self.paper_id = eval_config['ids']
            logger.info(f"Using specific paper IDs from config: {self.paper_id}")
        # Otherwise, paper_id will be set in load_data method

    def load_model(self):
        """Load trained model and tokenizer, downloading from Hugging Face if not present locally"""

        # Load pretrained model from directory
        if self.config['model']['trained']:
            self.load_model_from_dir(f'./Models/{self.config['model']['name']}-{self.dir_name}')

        # Download model
        else:
            self.initialize_model()

        self.model.to(self.device)


    def load_data(self):
        """Load evaluation dataset from JSON files based on config"""
        data_dir = self._get_data_directory()
        eval_path = os.path.join(data_dir, "eval.json")

        logger.info(f"Loading evaluation dataset from {data_dir}...")
        dataset = load_dataset(
            'json',
            data_files={'eval': eval_path},
            streaming=False  # Changed to non-streaming to support random selection and slicing
        )

        eval_dataset = dataset['eval']
        eval_config = self.config.get('eval', {})

        # If paper_id is already set (from ids in config), use those specific papers
        if self.paper_id is not None:
            # Convert paper_ids to strings if they're stored as integers in the JSON
            wanted_paper_ids = set(str(pid) for pid in self.paper_id)
            logger.info(f"Filtering dataset for specific paper IDs: {wanted_paper_ids}")

            # Filter the dataset
            self.dataset = eval_dataset.filter(
                lambda example: example['paper_id'] in wanted_paper_ids
            )
        else:
            # Get size from config
            size = eval_config.get('size', 10)

            # Check if random selection is enabled
            if eval_config.get('random', False):
                logger.info(f"Randomly selecting {size} papers for evaluation")
                # Shuffle the dataset and select the first 'size' papers
                shuffled_dataset = eval_dataset.shuffle(seed=42)
                self.dataset = shuffled_dataset.select(range(min(size, len(shuffled_dataset))))

                # Store the selected paper IDs
                self.paper_id = [example['paper_id'] for example in self.dataset]
            else:
                logger.info(f"Selecting first {size} papers for evaluation")
                # Select the first 'size' papers
                self.dataset = eval_dataset.select(range(min(size, len(eval_dataset))))

                # Store the selected paper IDs
                self.paper_id = [example['paper_id'] for example in self.dataset]

        logger.info(f"Selected {len(self.dataset)} papers for evaluation")
        return self.dataset

    def prepare_dataset(self):
        """Apply preprocessing and tokenization to evaluation dataset"""
        logger.info("Preprocessing evaluation dataset...")
        processed_dataset = self.dataset.map(
            self.preprocess,
            batched=True,
            remove_columns=['abstract', 'text', 'paper_id', 'title', 'section_names', 'domain']
        )

        self.eval_dataset = processed_dataset
        return self.eval_dataset

    def generate_summaries(self):
        self.model.eval()

        # Create a mapping of paper_id to generated output
        generated_outputs = {}

        # Single pass: generate outputs and update dataset in one iteration
        updated_dataset = []

        for i, text in enumerate(self.eval_dataset):
            input_text = text['input_text']

            # Use the generate_output function
            output = self.generate_output(
                input_text=input_text,
                max_length=150,  # Reduced from 5000 to avoid integer overflow
                num_beams=4,
                temperature=1.0
            )

            # Check if i is within the bounds of self.paper_id
            if i < len(self.paper_id):
                paper_id = self.paper_id[i]
                logger.info(f"Generated summary for paper {paper_id}: {output}")
                # Store the output in our map
                generated_outputs[paper_id] = output
            else:
                logger.info(f"Generated summary for example {i}: {output}")

            # Update the example with the output
            text['output_text'] = output
            updated_dataset.append(text)

        # Replace the dataset with the updated one
        self.eval_dataset = updated_dataset

        self.generated_outputs = generated_outputs
        return generated_outputs

    def evaluate_test(self):
        """Evaluate the model on the test dataset"""
        logger.info("Setting up evaluation...")

        # Extract target_text and output_text from the dataset using list comprehensions
        target_texts = [example['target_text'] for example in self.eval_dataset]
        output_texts = [example['output_text'] for example in self.eval_dataset]

        # Compute metrics using the compute_metrics function
        logger.info("Computing metrics for generated summaries...")
        metrics = self.compute_metrics(target_texts, output_texts)
        logger.info("Metrics computation completed.")

        self.eval_results = metrics
        return metrics

    def _convert_numpy_to_python(self, obj):
        """Convert NumPy types to Python types for JSON serialization"""

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def save_results(self):
        logger.info("Saving results and evaluation dataset...")

        # Ensure eval_dataset is in a serializable format (convert to list if it's an iterable)
        eval_dataset_list = list(self.eval_dataset) if hasattr(self.eval_dataset, '__iter__') else self.eval_dataset

        # Convert NumPy types to Python types for JSON serialization
        eval_results_serializable = self._convert_numpy_to_python(self.eval_results)
        eval_dataset_serializable = self._convert_numpy_to_python(eval_dataset_list)

        # Create a dictionary containing both eval results and dataset
        save_data = {
            "eval_results": eval_results_serializable,
            "eval_dataset": eval_dataset_serializable
        }

        # Ensure results directory exists
        results_dir = "./results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            logger.info(f"Created results directory: {results_dir}")

        # Save the combined data to a JSON file
        results_file = f"{results_dir}/{self.config['model']['name']}-{self.dir_name}.json"
        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Results and evaluation dataset saved successfully to {results_file}")



def main():
    """Main function to run the tester"""
    logger.info("Initializing TestModel...")
    t = TestModel()

    logger.info("Loading data...")
    t.load_data()

    logger.info("Loading model...")
    t.load_model()

    logger.info("Preparing dataset...")
    t.prepare_dataset()

    logger.info("Generating summaries...")
    summ = t.generate_summaries()

    logger.info("Evaluating test...")
    t.evaluate_test()

    logger.info("Saving results...")
    t.save_results()  # Save both evaluation results and dataset

    logger.info("Testing completed successfully!")


if __name__ == "__main__":
    main()
