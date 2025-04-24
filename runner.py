"""
The main purpose of this file is to be used as a runner for the server.
Merging the two process reducing compute time.
"""
import logging
import sys

from train import TrainingSummarization
from evaluate import EvaluationSummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logging goes to stdout for proper line updates
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize
        trainer = TrainingSummarization()
        evaluator = EvaluationSummarizationPipeline()

        # Load data
        trainer.load_data()

        # Initialize model components
        trainer.initialize_model()

        # Prepare datasets
        trainer.prepare_datasets()

        # Train
        t_results = trainer.train()

        # Save model
        trainer.save_model()

        # Configure evaluator
        evaluator.model = trainer.model
        evaluator.dataset = trainer.dataset['test']

        # Evaluate model
        e_results = evaluator.evaluate()

        logger.info(f'\nTraining results for model {trainer.config['model']['name']}:')
        logger.info(t_results)

        logger.info(f'\nTraining results for model {evaluator.config['model']['name']}:')
        logger.info(e_results)

    except Exception as e:
        logger.error(f"Py runner execution failed: {str(e)}")