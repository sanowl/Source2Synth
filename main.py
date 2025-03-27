import asyncio
import aiohttp
import async_lru
import numpy as np
import pandas as pd
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.asyncio import tqdm_asyncio
from functools import lru_cache
import spacy
from dotenv import load_dotenv
import secrets

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()  # Load environment variables from a .env file if present

# ============================================================
# Configuration and Logging Setup
# ============================================================

@dataclass
class Source2SynthConfig:
    dataset_size: int = 10000
    curation_ratio: float = 0.8
    model_name: str = "gpt2-large"
    max_length: int = 512
    temperature: float = 0.7
    num_beams: int = 4
    device: str = field(init=False)
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    seed: int = 42
    api_key: str = field(default_factory=lambda: os.getenv("WIKIPEDIA_API_KEY"))
    nlp_model_name: str = "en_core_web_sm"
    retry_attempts: int = 3
    retry_delay: float = 1.0  # in seconds

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        if not self.api_key:
            raise ValueError("WIKIPEDIA_API_KEY environment variable not set")

# Configure logging with detailed format and multiple handlers
def configure_logging(log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger('Source2Synth')
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler with rotation
    fh = logging.handlers.RotatingFileHandler(
        'source2synth.log', maxBytes=10**7, backupCount=5
    )
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = configure_logging()

# ============================================================
# Custom Exceptions
# ============================================================

class DataSourceError(Exception):
    """Exception raised when data source fails."""
    pass

class SeedGenerationError(Exception):
    """Exception raised during seed generation."""
    pass

class ExampleConstructionError(Exception):
    """Exception raised during example construction."""
    pass

class DataCurationError(Exception):
    """Exception raised during data curation."""
    pass

# ============================================================
# Abstract Base Classes
# ============================================================

class DataSource(ABC):
    @abstractmethod
    async def get_data(self, dataset_size: int) -> List[Dict[str, Any]]:
        """Retrieve data from the source asynchronously."""
        pass

class SeedGenerator(ABC):
    @abstractmethod
    def generate_seed(self, data: Dict[str, Any]) -> Optional[str]:
        """Generate a seed from the given data."""
        pass

class ExampleConstructor(ABC):
    @abstractmethod
    def construct_example(self, data: Dict[str, Any], seed: str) -> Dict[str, Any]:
        """Construct an example from the data and seed."""
        pass

class DataCurator(ABC):
    @abstractmethod
    def curate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Curate the dataset."""
        pass

# ============================================================
# Concrete Implementations
# ============================================================

class WikipediaDataSource(DataSource):
    """DataSource implementation for fetching data from Wikipedia asynchronously."""

    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None, retry_attempts: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    async def get_data(self, dataset_size: int) -> List[Dict[str, Any]]:
        """Asynchronously retrieve data from Wikipedia."""
        self.logger.info("Starting data retrieval from Wikipedia.")
        articles = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_random_article(session)
                for _ in range(dataset_size)
            ]
            for future in tqdm_asyncio.as_completed(tasks, total=dataset_size, desc="Fetching articles"):
                try:
                    article = await future
                    if article:
                        full_article = await self.fetch_full_article(session, article['pageid'])
                        if full_article:
                            articles.append(full_article)
                        if len(articles) >= dataset_size:
                            break
                except DataSourceError as e:
                    self.logger.error(f"Data source error: {e}")
        self.logger.info(f"Fetched {len(articles)} articles from Wikipedia.")
        return articles

    @async_lru.alru_cache(maxsize=1000)
    async def fetch_random_article(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch a single random article from Wikipedia with retry mechanism."""
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": 1
        }
        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f"Non-200 response: {response.status}. Attempt {attempt} of {self.retry_attempts}.")
                        await asyncio.sleep(self.retry_delay)
                        continue
                    data = await response.json()
                    random_page = data.get("query", {}).get("random", [{}])[0]
                    title = random_page.get("title")
                    pageid = random_page.get("id")
                    if title and pageid:
                        return {"title": title, "pageid": pageid}
                    self.logger.debug("Incomplete article data received.")
                    return None
            except aiohttp.ClientError as e:
                self.logger.warning(f"Client error: {e}. Attempt {attempt} of {self.retry_attempts}.")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}. Attempt {attempt} of {self.retry_attempts}.")
                await asyncio.sleep(self.retry_delay)
        self.logger.error("Max retry attempts reached. Failed to fetch article.")
        return None

    async def fetch_full_article(self, session: aiohttp.ClientSession, pageid: int) -> Optional[Dict[str, Any]]:
        """Fetch the full text of a Wikipedia article using its pageid."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,
            "pageids": pageid,
            "redirects": 1
        }
        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f"Non-200 response while fetching full article: {response.status}. Attempt {attempt} of {self.retry_attempts}.")
                        await asyncio.sleep(self.retry_delay)
                        continue
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    page = pages.get(str(pageid), {})
                    extract = page.get("extract", "")
                    if extract:
                        return {"text": extract, "related_texts": self.extract_related_texts(extract)}
                    self.logger.debug("No extract found in article.")
                    return None
            except aiohttp.ClientError as e:
                self.logger.warning(f"Client error while fetching full article: {e}. Attempt {attempt} of {self.retry_attempts}.")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error while fetching full article: {e}. Attempt {attempt} of {self.retry_attempts}.")
                await asyncio.sleep(self.retry_delay)
        self.logger.error("Max retry attempts reached. Failed to fetch full article.")
        return None

    def extract_related_texts(self, text: str) -> List[str]:
        """
        Extract related texts from the article.
        For simplicity, we'll extract the first few sentences as related texts.
        This can be enhanced to extract more meaningful related texts.
        """
        sentences = text.split('. ')
        related_texts = sentences[1:4] if len(sentences) >=4 else sentences[1:]
        return related_texts

class EntitySeedGenerator(SeedGenerator):
    """SeedGenerator implementation using spaCy NLP model to extract entities."""

    def __init__(self, nlp_model, logger: Optional[logging.Logger] = None):
        self.nlp_model = nlp_model
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def generate_seed(self, data: Dict[str, Any]) -> Optional[str]:
        """Generate a seed entity from the given data."""
        try:
            text = data.get('text', '')
            if not text:
                self.logger.debug("Empty text provided for seed generation.")
                return None
            doc = self.nlp_model(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT"}]
            if not entities:
                self.logger.debug("No relevant entities found in text.")
                return None
            seed = secrets.choice(entities)
            self.logger.debug(f"Generated seed: {seed}")
            return seed
        except Exception as e:
            self.logger.error(f"Error generating seed: {e}")
            raise SeedGenerationError(f"Failed to generate seed: {e}") from e

class MHQAExampleConstructor(ExampleConstructor):
    """ExampleConstructor implementation for Multi-Hop QA."""

    def __init__(self, config: Source2SynthConfig, tokenizer, model, logger: Optional[logging.Logger] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def construct_example(self, data: Dict[str, Any], seed: str) -> Dict[str, Any]:
        """Construct a multi-hop QA example from data and seed."""
        try:
            context = data.get('text', '')
            related_texts = data.get('related_texts', [])
            if not context:
                raise ExampleConstructionError("Context text is empty.")
            q1 = self._generate_question(context, seed, "Q1")
            q2 = self._generate_question(" ".join(related_texts), seed, "Q2")
            merged_q = self._merge_questions(q1, q2)
            self.logger.debug(f"Constructed example with merged question: {merged_q}")
            return {
                "question": merged_q,
                "q1": q1,
                "q2": q2,
                "seed": seed,
                "context": context
            }
        except Exception as e:
            self.logger.error(f"Error constructing example: {e}")
            raise ExampleConstructionError(f"Failed to construct example: {e}") from e

    def _generate_question(self, text: str, seed: str, q_label: str) -> str:
        """Generate a question based on text and seed."""
        prompt = f"Generate a {q_label} based on the following text where the answer is '{seed}': {text}"
        return self._generate_text(prompt)

    def _merge_questions(self, q1: str, q2: str) -> str:
        """Merge two questions into a single multi-hop question."""
        prompt = f"Merge these two questions into a single coherent multi-hop question.\nQ1: {q1}\nQ2: {q2}"
        return self._generate_text(prompt)

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the language model based on the prompt."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.config.max_length, truncation=True)
            inputs = inputs.to(self.config.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=self.config.max_length,
                    num_return_sequences=1,
                    temperature=self.config.temperature,
                    num_beams=self.config.num_beams,
                    early_stopping=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.debug(f"Generated text: {generated_text}")
            return generated_text.strip()
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise ExampleConstructionError(f"Failed to generate text: {e}") from e

class MLFlowDataCurator(DataCurator):
    """DataCurator implementation using MLflow for experiment tracking."""

    def __init__(self, config: Source2SynthConfig, model, logger: Optional[logging.Logger] = None):
        self.config = config
        self.model = model
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def curate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Curate the dataset and log metrics to MLflow."""
        try:
            self.logger.info("Starting data curation.")
            mlflow.start_run()
            mlflow.log_params(OmegaConf.to_container(OmegaConf.structured(self.config), resolve=True))
            curated_dataset = [example for example in dataset if self._is_high_quality(example)]
            curation_rate = len(curated_dataset) / len(dataset) if dataset else 0
            self.logger.info(f"Curation complete. Curation rate: {curation_rate:.2f}")
            mlflow.log_metric("curation_rate", curation_rate)
            mlflow.end_run()
            return curated_dataset
        except Exception as e:
            self.logger.error(f"Error during data curation: {e}")
            mlflow.end_run()
            raise DataCurationError(f"Failed to curate data: {e}") from e

    def _is_high_quality(self, example: Dict[str, Any]) -> bool:
        """Determine if an example is of high quality."""
        question = example.get("question", "")
        context = example.get("context", "")
        seed = example.get("seed", "")
        if not question or not context or not seed:
            self.logger.debug("Example missing question, context, or seed.")
            return False
        if len(question.split()) < 5 or len(context.split()) < 50:
            self.logger.debug("Question or context too short.")
            return False
        if seed not in context:
            self.logger.debug("Seed not found in context.")
            return False
        self.logger.debug("Example quality: High")
        return True

# ============================================================
# Custom PyTorch Dataset
# ============================================================

class Source2SynthDataset(Dataset):
    """Custom PyTorch Dataset for Source2Synth."""

    def __init__(self, examples: List[Dict[str, Any]], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        question = example.get('question', '')
        context = example.get('context', '')
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ============================================================
# Main Source2Synth Class
# ============================================================

class Source2Synth:
    """Main class orchestrating data generation, curation, and model training."""

    def __init__(self, config: Source2SynthConfig):
        self.config = config
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.config.device)
        self.data_source = WikipediaDataSource(
            api_key=self.config.api_key,
            logger=self.logger,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay
        )
        self.seed_generator = EntitySeedGenerator(
            nlp_model=self._load_nlp_model(),
            logger=self.logger
        )
        self.example_constructor = MHQAExampleConstructor(
            config=self.config,
            tokenizer=self.tokenizer,
            model=self.model,
            logger=self.logger
        )
        self.data_curator = MLFlowDataCurator(
            config=self.config,
            model=self.model,
            logger=self.logger
        )

    def _load_nlp_model(self):
        """Load the NLP model for seed generation."""
        try:
            nlp = spacy.load(self.config.nlp_model_name)
            self.logger.info(f"NLP model '{self.config.nlp_model_name}' loaded successfully.")
            return nlp
        except Exception as e:
            self.logger.error(f"Failed to load NLP model '{self.config.nlp_model_name}': {e}")
            raise SeedGenerationError(f"Cannot load NLP model: {e}") from e

    async def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate dataset by retrieving data, generating seeds, and constructing examples."""
        self.logger.info("Generating dataset.")
        raw_data = await self.data_source.get_data(self.config.dataset_size)
        dataset = []
        for data in tqdm_asyncio(raw_data, desc="Processing data"):
            seed = self.seed_generator.generate_seed(data)
            if not seed:
                self.logger.debug("No seed generated; skipping example.")
                continue
            try:
                example = self.example_constructor.construct_example(data, seed)
                dataset.append(example)
            except ExampleConstructionError as e:
                self.logger.debug(f"Skipping example due to error: {e}")
                continue
        self.logger.info(f"Generated {len(dataset)} examples.")
        return dataset

    def curate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Curate the generated dataset."""
        self.logger.info("Curating dataset.")
        curated = self.data_curator.curate(dataset)
        self.logger.info(f"Curated dataset contains {len(curated)} examples.")
        return curated

    def train_model(self, dataset: List[Dict[str, Any]]):
        """Train the language model using the curated dataset."""
        self.logger.info("Starting model training.")
        train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=self.config.seed)
        train_dataset = Source2SynthDataset(train_data, self.tokenizer)
        val_dataset = Source2SynthDataset(val_data, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            seed=self.config.seed,
            report_to=["mlflow"],
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        trainer.save_model("./source2synth_model")
        self.logger.info("Model training completed and saved.")

# ============================================================
# Main Function with Hydra Configuration
# ============================================================

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    """Main entry point for the Source2Synth pipeline."""
    try:
        # Convert DictConfig to Source2SynthConfig dataclass
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config = Source2SynthConfig(**config_dict)

        # Initialize Source2Synth
        source2synth = Source2Synth(config)

        # Generate Dataset
        dataset = asyncio.run(source2synth.generate_dataset())

        # Curate Dataset
        curated_dataset = source2synth.curate_dataset(dataset)

        # Train Model
        source2synth.train_model(curated_dataset)

        logger.info("Source2Synth process completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during the Source2Synth process: {e}")
        sys.exit(1)

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # Example configuration for Hydra
    example_config = {
        "dataset_size": 5000,
        "curation_ratio": 0.85,
        "model_name": "gpt2-medium",
        "max_length": 256,
        "temperature": 0.8,
        "num_beams": 5,
        "learning_rate": 3e-5,
        "batch_size": 8,
        "num_epochs": 5,
        "seed": 123,
        "nlp_model_name": "en_core_web_sm",
        "retry_attempts": 5,
        "retry_delay": 2.0
    }
    # Convert example_config to DictConfig
    cfg = OmegaConf.create(example_config)
    main(cfg)
