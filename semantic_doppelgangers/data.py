"""
Data loading and preprocessing functionality for semantic doppelgängers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class CorpusLoader:
    """
    Handles loading and preprocessing text data from various sources.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the corpus loader.

        Args:
            cache_dir: Optional directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)

    def load_opensubtitles(
        self,
        languages: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from OpenSubtitles dataset.

        Args:
            languages: List of language codes to load
            max_samples: Maximum number of samples per language

        Returns:
            Tuple of (texts, languages, sources)
        """
        dataset = load_dataset(
            "opensubtitles",
            cache_dir=self.cache_dir,
            languages=languages or ["en"],
        )

        texts = []
        langs = []
        sources = []

        for lang in dataset.keys():
            for item in tqdm(
                dataset[lang]["train"],
                desc=f"Loading {lang}",
                total=min(max_samples or float("inf"), len(dataset[lang]["train"])),
            ):
                if max_samples and len(texts) >= max_samples:
                    break
                    
                if item["text"].strip():
                    texts.append(item["text"])
                    langs.append(lang)
                    sources.append(f"opensubtitles_{item['id']}")

        return texts, langs, sources

    def load_wikipedia(
        self,
        languages: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from Wikipedia dataset.

        Args:
            languages: List of language codes to load
            max_samples: Maximum number of samples per language

        Returns:
            Tuple of (texts, languages, sources)
        """
        dataset = load_dataset(
            "wikipedia",
            cache_dir=self.cache_dir,
            languages=languages or ["en"],
        )

        texts = []
        langs = []
        sources = []

        for lang in dataset.keys():
            for item in tqdm(
                dataset[lang]["train"],
                desc=f"Loading {lang}",
                total=min(max_samples or float("inf"), len(dataset[lang]["train"])),
            ):
                if max_samples and len(texts) >= max_samples:
                    break
                    
                if item["text"].strip():
                    texts.append(item["text"])
                    langs.append(lang)
                    sources.append(f"wikipedia_{item['id']}")

        return texts, langs, sources

    def load_reddit(
        self,
        subreddits: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load data from Reddit dataset.

        Args:
            subreddits: List of subreddits to load
            max_samples: Maximum number of samples per subreddit

        Returns:
            Tuple of (texts, languages, sources)
        """
        dataset = load_dataset(
            "reddit",
            cache_dir=self.cache_dir,
            subreddits=subreddits or ["all"],
        )

        texts = []
        langs = []
        sources = []

        for item in tqdm(
            dataset["train"],
            desc="Loading Reddit",
            total=min(max_samples or float("inf"), len(dataset["train"])),
        ):
            if max_samples and len(texts) >= max_samples:
                break
                
            if item["text"].strip():
                texts.append(item["text"])
                langs.append("en")  # Assuming English for now
                sources.append(f"reddit_{item['id']}")

        return texts, langs, sources

    def load_custom(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        language_column: Optional[str] = None,
        source_column: Optional[str] = None,
        domain_column: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load data from a custom file (CSV, JSON, or JSONL).

        Args:
            file_path: Path to the data file
            text_column: Name of the column containing text
            language_column: Optional name of the language column
            source_column: Optional name of the source column
            domain_column: Optional name of the domain column

        Returns:
            Tuple of (texts, languages, sources, domains)
        """
        file_path = Path(file_path)
        
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        texts = df[text_column].tolist()
        languages = df[language_column].tolist() if language_column else ["unknown"] * len(texts)
        sources = df[source_column].tolist() if source_column else ["custom"] * len(texts)
        domains = df[domain_column].tolist() if domain_column else ["unknown"] * len(texts)

        return texts, languages, sources, domains

    def preprocess_text(
        self,
        text: str,
        min_length: int = 10,
        max_length: int = 512,
    ) -> Optional[str]:
        """
        Preprocess a single text.

        Args:
            text: Text to preprocess
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            Preprocessed text or None if text should be filtered out
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove URLs
        text = " ".join(word for word in text.split() if not word.startswith(("http://", "https://")))
        
        # Remove special characters
        text = " ".join(word for word in text.split() if word.isalnum() or word.isspace())
        
        # Length filtering
        if len(text) < min_length or len(text) > max_length:
            return None
            
        return text

    def preprocess_corpus(
        self,
        texts: List[str],
        languages: List[str],
        sources: List[str],
        domains: Optional[List[str]] = None,
        min_length: int = 10,
        max_length: int = 512,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Preprocess a corpus of texts.

        Args:
            texts: List of texts
            languages: List of languages
            sources: List of sources
            domains: Optional list of domains
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            Tuple of preprocessed (texts, languages, sources, domains)
        """
        if domains is None:
            domains = ["unknown"] * len(texts)

        processed_texts = []
        processed_langs = []
        processed_sources = []
        processed_domains = []

        for text, lang, source, domain in tqdm(
            zip(texts, languages, sources, domains),
            desc="Preprocessing corpus",
            total=len(texts),
        ):
            processed_text = self.preprocess_text(text, min_length, max_length)
            if processed_text:
                processed_texts.append(processed_text)
                processed_langs.append(lang)
                processed_sources.append(source)
                processed_domains.append(domain)

        return processed_texts, processed_langs, processed_sources, processed_domains 