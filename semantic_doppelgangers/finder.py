"""
Core functionality for finding semantic doppelgängers across languages and domains.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from semantic_doppelgangers.models import MultilingualEmbedder
from semantic_doppelgangers.linguistic.analysis import LinguisticAnalyzer, LinguisticFeatures
from semantic_doppelgangers.linguistic.advanced_analysis import AdvancedLinguisticAnalyzer, AdvancedLinguisticFeatures


@dataclass
class SearchResult:
    """Represents a single search result with similarity score and metadata."""
    text: str
    similarity: float
    source: str
    language: str
    domain: str
    linguistic_features: Optional[LinguisticFeatures] = None
    advanced_features: Optional[AdvancedLinguisticFeatures] = None


class DoppelgangerFinder:
    """
    Main class for finding semantic doppelgängers across languages and domains.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        enable_linguistic_features: bool = True,
        enable_advanced_analysis: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the doppelgänger finder.

        Args:
            model_name: Name of the sentence transformer model
            enable_linguistic_features: Whether to enable linguistic analysis
            enable_advanced_analysis: Whether to enable advanced linguistic analysis
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize FAISS index
        self.dimension = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Storage for metadata
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.languages: List[str] = []
        self.domains: List[str] = []
        
        # Initialize linguistic analyzers if enabled
        self.linguistic_analyzer = None
        self.advanced_analyzer = None
        
        if enable_linguistic_features:
            self.linguistic_analyzer = LinguisticAnalyzer()
            
        if enable_advanced_analysis:
            self.advanced_analyzer = AdvancedLinguisticAnalyzer()

    def add_corpus(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        batch_size: int = 32,
    ):
        """
        Add a corpus of texts to the search index.

        Args:
            texts: List of texts to add
            sources: Optional list of source identifiers
            languages: Optional list of language codes
            domains: Optional list of domain identifiers
            batch_size: Batch size for embedding computation
        """
        # Generate embeddings
        embeddings = self.model.encode(texts, batch_size=batch_size)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.texts.extend(texts)
        self.sources.extend(sources or ["unknown"] * len(texts))
        self.languages.extend(languages or ["unknown"] * len(texts))
        self.domains.extend(domains or ["unknown"] * len(texts))

    def find_doppelgangers(
        self,
        query: str,
        top_k: int = 5,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        min_similarity: float = 0.0,
        analyze_linguistic_features: bool = True,
        analyze_advanced_features: bool = True,
        reference_texts: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Find semantic doppelgängers for a given query.

        Args:
            query: Query text
            top_k: Number of results to return
            domains: Optional list of domains to search in
            languages: Optional list of languages to search in
            min_similarity: Minimum similarity threshold
            analyze_linguistic_features: Whether to analyze linguistic features
            analyze_advanced_features: Whether to perform advanced analysis
            reference_texts: Optional reference texts for advanced analysis

        Returns:
            List of SearchResult objects
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Search in index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k=min(top_k * 2, len(self.texts))  # Get more results for filtering
        )
        
        # Convert distances to similarities (if using L2)
        similarities = 1 / (1 + distances[0])
        
        # Filter and format results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity < min_similarity:
                continue
                
            if domains and self.domains[idx] not in domains:
                continue
                
            if languages and self.languages[idx] not in languages:
                continue
            
            # Analyze linguistic features if enabled
            linguistic_features = None
            if (analyze_linguistic_features and self.linguistic_analyzer):
                linguistic_features = self.linguistic_analyzer.analyze_text(
                    self.texts[idx],
                    language=self.languages[idx],
                )
                
            # Perform advanced analysis if enabled
            advanced_features = None
            if (analyze_advanced_features and self.advanced_analyzer):
                advanced_features = self.advanced_analyzer.analyze_text(
                    self.texts[idx],
                    language=self.languages[idx],
                    reference_texts=reference_texts,
                )
                
            results.append(
                SearchResult(
                    text=self.texts[idx],
                    similarity=float(similarity),
                    source=self.sources[idx],
                    language=self.languages[idx],
                    domain=self.domains[idx],
                    linguistic_features=linguistic_features,
                    advanced_features=advanced_features,
                )
            )
            
            if len(results) >= top_k:
                break
                
        return results

    def save_index(self, path: str):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, f"{path}/index.faiss")
        np.save(f"{path}/metadata.npz", {
            "texts": self.texts,
            "sources": self.sources,
            "languages": self.languages,
            "domains": self.domains,
        })

    def load_index(self, path: str):
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(f"{path}/index.faiss")
        metadata = np.load(f"{path}/metadata.npz", allow_pickle=True)
        self.texts = metadata["texts"].tolist()
        self.sources = metadata["sources"].tolist()
        self.languages = metadata["languages"].tolist()
        self.domains = metadata["domains"].tolist()

    def analyze_doppelganger_pair(
        self,
        text1: str,
        text2: str,
        language1: str = "en",
        language2: str = "en",
        reference_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Analyze a pair of texts for doppelgänger characteristics.

        Args:
            text1: First text
            text2: Second text
            language1: Language of first text
            language2: Language of second text
            reference_texts: Optional reference texts for advanced analysis
            
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        # Basic similarity
        emb1 = self.model.encode([text1])[0]
        emb2 = self.model.encode([text2])[0]
        similarity = 1 - np.linalg.norm(emb1 - emb2) / 2
        results["semantic_similarity"] = float(similarity)
        
        # Linguistic analysis
        if self.linguistic_analyzer:
            features1 = self.linguistic_analyzer.analyze_text(text1, language=language1)
            features2 = self.linguistic_analyzer.analyze_text(text2, language=language2)
            
            results["linguistic_analysis"] = {
                "text1": {
                    "style": features1.style,
                    "register": features1.register,
                    "syntactic_complexity": features1.syntactic_complexity,
                    "semantic_density": features1.semantic_density,
                },
                "text2": {
                    "style": features2.style,
                    "register": features2.register,
                    "syntactic_complexity": features2.syntactic_complexity,
                    "semantic_density": features2.semantic_density,
                },
            }
            
        # Advanced analysis
        if self.advanced_analyzer:
            adv_features1 = self.advanced_analyzer.analyze_text(
                text1,
                language=language1,
                reference_texts=reference_texts,
            )
            adv_features2 = self.advanced_analyzer.analyze_text(
                text2,
                language=language2,
                reference_texts=reference_texts,
            )
            
            results["advanced_analysis"] = {
                "text1": {
                    "dependency_tree_depth": adv_features1.dependency_tree_depth,
                    "clause_complexity": adv_features1.clause_complexity,
                    "lexical_diversity": adv_features1.lexical_diversity,
                    "coherence_score": adv_features1.coherence_score,
                    "entropy": adv_features1.entropy,
                    "information_density": adv_features1.information_density,
                },
                "text2": {
                    "dependency_tree_depth": adv_features2.dependency_tree_depth,
                    "clause_complexity": adv_features2.clause_complexity,
                    "lexical_diversity": adv_features2.lexical_diversity,
                    "coherence_score": adv_features2.coherence_score,
                    "entropy": adv_features2.entropy,
                    "information_density": adv_features2.information_density,
                },
            }
            
        return results 