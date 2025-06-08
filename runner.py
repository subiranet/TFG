"""
Runner script for the summarization pipeline.

This script orchestrates the complete pipeline process:
1. Training a summarization model
2. Testing the trained model on test data
3. Evaluating and saving results

By combining these processes in a single script, it reduces overall compute time
and ensures consistent model usage across stages.
"""
import logging
import sys

from train import TrainingSummarization
from evaluate import EvaluationSummarizationPipeline
from tester import TestModel

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logging goes to stdout for proper line updates
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the runner script.

    Orchestrates the complete pipeline:
    1. Initializes training, evaluation, and testing components
    2. Loads and prepares data
    3. Trains the model
    4. Tests the trained model
    5. Evaluates and saves results
    """
    # Initialize components
    trainer = TrainingSummarization()
    evaluator = EvaluationSummarizationPipeline()
    tester = TestModel()

    # Load data
    trainer.load_data()

    # Initialize model components
    trainer.initialize_model()

    # Prepare datasets
    trainer.prepare_datasets()

    # Train
    t_results = trainer.train()

    # Uncomment to save the model to disk if needed
    # trainer.save_model()

    # Uncomment for separate evaluation if needed
    # evaluator.model = trainer.model
    # evaluator.dataset = trainer.dataset['test']
    # e_results = evaluator.evaluate()
    # logger.info(f'\nEvaluation results for model {evaluator.config['model']['name']}:')
    # logger.info(e_results)

    # Run testing on the trained model
    logger.info("Loading data...")
    tester.load_data()

    logger.info("Loading model...")
    tester.model = trainer.model
    tester.tokenizer = trainer.tokenizer

    logger.info("Preparing dataset...")
    tester.prepare_dataset()

    logger.info("Generating summaries...")
    summ = tester.generate_summaries()

    logger.info("Evaluating test...")
    tester.evaluate_test()

    logger.info("Saving results...")
    tester.save_results()  # Save both evaluation results and dataset

    # Log training results
    logger.info(f'\nTraining results for model {trainer.config["model"]["name"]}:')
    logger.info(t_results)

    logger.info("Testing completed successfully!")




if __name__ == "__main__":
    main()
