import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any


def get_project_dirs() -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir.parent
    config_dir = data_dir.parent
    store_dir = data_dir / "Treated"
    return script_dir, data_dir, config_dir, store_dir


def validate_ratios(ratios: Dict[str, float]) -> None:
    if not 0.99 <= ratios['train'] + ratios['test'] + ratios['eval'] <= 1.01:
        print(ratios.values())
        raise ValueError("Ratios must sum to approximately 1 (0.99-1.01)")


def split_data(data: List[Any], ratios: Dict[str, float], config_data: Dict[str, Any] = None) -> Tuple[List[Any], List[Any], List[Any]]:
    validate_ratios(ratios)

    # Check if we have specific IDs for evaluation dataset
    eval_ids = config_data.get('eval', {}).get('ids', []) if config_data else []

    # If we have specific IDs, filter them out first
    eval_data = []
    remaining_data = []

    if eval_ids:
        # Convert all IDs to strings for consistent comparison
        eval_ids = [str(id_) for id_ in eval_ids]

        # First pass: collect papers with matching IDs for eval dataset
        for paper in data:
            if str(paper.get('paper_id', '')) in eval_ids:
                eval_data.append(paper)
            else:
                remaining_data.append(paper)

        print(f"Found {len(eval_data)} papers matching the specified IDs for evaluation")

        # If we couldn't find all the specified papers, log a warning
        if len(eval_data) < len(eval_ids):
            print(f"Warning: Could only find {len(eval_data)} out of {len(eval_ids)} specified papers")

        # Shuffle the remaining data for random train/test split
        random.shuffle(remaining_data)

        # Limit to the total number specified in config
        total_needed = ratios['total']
        eval_count = len(eval_data)
        remaining_needed = total_needed - eval_count

        if remaining_needed <= 0:
            print(f"Warning: The number of eval papers ({eval_count}) exceeds or equals the total required ({total_needed})")
            # In this case, we'll just use the eval papers and have empty train/test sets
            return [], [], eval_data[:total_needed]

        remaining_data = remaining_data[:remaining_needed]

        # Calculate train/test split based on the remaining data
        train_ratio = ratios['train'] / (ratios['train'] + ratios['test'])
        train_end = int(len(remaining_data) * train_ratio)

        train_data = remaining_data[:train_end]
        test_data = remaining_data[train_end:]
    else:
        # Original behavior if no specific IDs are provided
        random.shuffle(data)
        data = data[:ratios['total']]

        train_end = int(ratios['total'] * ratios['train'])
        test_end = train_end + int(ratios['total'] * ratios['test'])

        train_data = data[:train_end]
        test_data = data[train_end:test_end]
        eval_data = data[test_end:]

    return train_data, test_data, eval_data


def save_data(data: List[Any], file_path: Path) -> None:
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def process_data() -> None:
    _, data_dir, config_dir, store_dir = get_project_dirs()

    config_file = config_dir / 'config.json'
    try:
        with open(config_file, 'r') as config:
            config_data = json.load(config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in config file at {config_file}")

    ratios = config_data['data']

    dir_name = (f"{int(100*ratios['train'])}-"
                f"{int(100*ratios['test'])}-"
                f"{int(100*ratios['eval'])}-"
                f"{int(ratios['total'])}")
    output_dir = store_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)


    print(f"Processing data and saving to {output_dir}")

    input_file = data_dir / 'papers.SSN.jsonl'
    try:
        with open(input_file, 'r') as infile:
            data = [json.loads(line) for line in infile]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input data file not found at {input_file}")

    train, test, eval_ = split_data(data, ratios, config_data)

    save_data(train, output_dir / 'train.json')
    save_data(test, output_dir / 'test.json')
    save_data(eval_, output_dir / 'eval.json')

    print(
        f"Data split and saved successfully to {output_dir}\n"
        f"Train: {len(train)} items\n"
        f"Test: {len(test)} items\n"
        f"Eval: {len(eval_)} items"
    )


if __name__ == "__main__":
    try:
        process_data()
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise
