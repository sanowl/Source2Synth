
# Source2Synth

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Logging & Monitoring](#logging--monitoring)
- [Model Training](#model-training)
- [Experiment Tracking](#experiment-tracking)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

**Source2Synth** is a comprehensive and scalable pipeline designed to generate, curate, and train machine learning models for Multi-Hop Question Answering (MHQA). Leveraging data from Wikipedia, advanced NLP techniques, and state-of-the-art language models, Source2Synth automates the creation of high-quality datasets and facilitates efficient model training.

## Features

- **Asynchronous Data Retrieval:** Efficiently fetches large-scale data from Wikipedia using `asyncio` and `aiohttp`.
- **Advanced Logging:** Structured logging with multiple handlers for comprehensive monitoring.
- **Robust Error Handling:** Custom exceptions and retry mechanisms ensure resilience against failures.
- **Caching Mechanism:** Reduces redundant API calls with `async_lru` caching.
- **Configuration Management:** Dynamic and flexible configurations using `Hydra`.
- **Data Validation:** Ensures data integrity and quality through rigorous validation checks.
- **Progress Tracking:** Real-time progress visualization with `tqdm`.
- **Dependency Injection:** Enhances flexibility and testability of components.
- **Optimized Data Processing:** Efficient handling of large datasets with optimized data structures.
- **Comprehensive Documentation:** Clear docstrings and type annotations for maintainability.
- **Environment Variable Management:** Secure handling of sensitive information using `.env` files.
- **Model Training Enhancements:** Advanced training techniques including mixed precision and early stopping.
- **Performance Monitoring:** Integration-ready with tools like `torch.profiler`.
- **Unit Testing Integration:** Modular design facilitates easy testing.
- **Resource Management:** Efficient handling of network sessions and model memory.

## Architecture

```
source2synth/
├── config/
│   └── config.yaml
├── data_sources/
│   └── wikipedia.py
├── generators/
│   └── seed_generator.py
├── constructors/
│   └── example_constructor.py
├── curators/
│   └── mlflow_curator.py
├── datasets/
│   └── source2synth_dataset.py
├── models/
│   └── source2synth_model.py
├── utils/
│   ├── logger.py
│   └── exceptions.py
├── tests/
│   └── test_components.py
└── main.py
```

## Installation

### Prerequisites

- **Python:** Version 3.8 or higher.
- **CUDA:** If you intend to utilize GPU acceleration, ensure that CUDA is installed and properly configured.

### Clone the Repository

```bash
git clone https://github.com/yourusername/source2synth.git
cd source2synth
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Ensure that you have the necessary build tools installed for packages like `torch`. Refer to [PyTorch Installation](https://pytorch.org/get-started/locally/) for detailed instructions.

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory to securely manage sensitive information like API keys.

```env
WIKIPEDIA_API_KEY=your_actual_api_key_here
```

> **Important:** Replace `your_actual_api_key_here` with your actual Wikipedia API key.

### Configuration File

Source2Synth utilizes `Hydra` for dynamic configuration management. The default configurations are defined within the script, but you can override them as needed.

Example configuration parameters:

```yaml
dataset_size: 5000
curation_ratio: 0.85
model_name: "gpt2-medium"
max_length: 256
temperature: 0.8
num_beams: 5
learning_rate: 3e-5
batch_size: 32
num_epochs: 5
seed: 123
api_key: "your_actual_api_key_here"
nlp_model_name: "en_core_web_sm"
retry_attempts: 5
retry_delay: 2.0
```

You can modify these parameters directly in the `main.py` or pass them via command-line arguments.

## Usage

### Running the Pipeline

Execute the main script to start the Source2Synth pipeline:

```bash
python source2synth.py
```

The pipeline will perform the following steps:

1. **Data Retrieval:** Fetches random articles from Wikipedia asynchronously.
2. **Seed Generation:** Extracts relevant entities using the spaCy NLP model.
3. **Example Construction:** Generates multi-hop QA examples based on the seeds.
4. **Data Curation:** Filters and curates the dataset based on quality metrics.
5. **Model Training:** Trains the language model using the curated dataset.

### Customizing Configurations

You can override default configurations using Hydra's command-line interface. For example:

```bash
python source2synth.py dataset_size=10000 batch_size=64
```

## Logging & Monitoring

Source2Synth employs advanced logging mechanisms to track the pipeline's progress and debug issues effectively.

- **Console Logs:** Real-time logs are displayed in the terminal.
- **File Logs:** Detailed logs are saved in `source2synth.log` with log rotation to manage file sizes.

### Log Levels

- **INFO:** General information about the pipeline's progress.
- **DEBUG:** Detailed diagnostic information.
- **WARNING:** Indications of potential issues.
- **ERROR:** Errors that prevent certain operations.

## Model Training

Source2Synth leverages the Hugging Face `transformers` library to train causal language models.

### Training Arguments

Key training parameters include:

- **num_train_epochs:** Number of training epochs.
- **learning_rate:** Learning rate for optimization.
- **batch_size:** Batch size for training and evaluation.
- **warmup_steps:** Number of warmup steps for learning rate scheduler.
- **weight_decay:** Weight decay for regularization.
- **fp16:** Enables mixed precision training for faster computation on compatible GPUs.
- **early_stopping_patience:** Number of evaluation steps with no improvement before stopping training.

### Training Process

The training process includes:

1. **Dataset Splitting:** Divides the curated dataset into training and validation sets.
2. **Dataset Preparation:** Utilizes a custom PyTorch `Dataset` for efficient data handling.
3. **Model Initialization:** Loads the specified language model and tokenizer.
4. **Training Execution:** Trains the model using the `Trainer` API with specified arguments.
5. **Model Saving:** Saves the trained model to the `./source2synth_model` directory.

## Experiment Tracking

Source2Synth integrates with **MLflow** for experiment tracking and management.

### Setting Up MLflow

1. **Install MLflow:**

   ```bash
   pip install mlflow
   ```

2. **Start MLflow Server:**

   ```bash
   mlflow ui
   ```

   Access the MLflow UI at [http://localhost:5000](http://localhost:5000).

### Tracking Experiments

During data curation and model training, Source2Synth logs parameters and metrics to MLflow, enabling you to:

- Compare different experiments.
- Visualize training metrics.
- Reproduce results.

## Testing

Source2Synth is designed with testability in mind, following a modular architecture that facilitates unit testing.

### Running Tests

Tests are located in the `tests/` directory. You can run them using `pytest`.

1. **Install Testing Dependencies:**

   ```bash
   pip install pytest
   ```

2. **Execute Tests:**

   ```bash
   pytest tests/
   ```

> **Note:** Ensure that your environment is correctly configured before running tests to avoid failures related to missing dependencies or configurations.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug reports, feel free to open an issue or submit a pull request.

### Guidelines

1. **Fork the Repository:** Create a personal fork of the project.
2. **Create a Branch:** Use descriptive branch names for your features or fixes.
3. **Commit Changes:** Follow conventional commit messages for clarity.
4. **Open a Pull Request:** Submit your changes for review.

### Code of Conduct

Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and respectful environment for all contributors.


## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/)
- [Hydra](https://hydra.cc/)
- [MLflow](https://mlflow.org/)
- [aiohttp](https://docs.aiohttp.org/)
- [tqdm](https://tqdm.github.io/)
- [async_lru](https://github.com/dnouri/async_lru)

---

*Happy Coding and Best of Luck with Your Source2Synth Pipeline! 🚀*