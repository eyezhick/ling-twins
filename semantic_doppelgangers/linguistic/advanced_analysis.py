"""
Advanced linguistic analysis functionality for semantic doppelgängers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import math
import numpy as np
from collections import Counter
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from spacy.tokens import Doc, Token
from spacy.language import Language


@dataclass
class AdvancedLinguisticFeatures:
    """Advanced linguistic features for research analysis."""
    # Syntactic features
    dependency_tree_depth: float
    clause_complexity: float
    phrase_structure_complexity: float
    
    # Semantic features
    lexical_diversity: float  # Type-token ratio
    semantic_field_density: Dict[str, float]  # Distribution across semantic fields
    conceptual_metaphors: List[str]
    
    # Discourse features
    coherence_score: float
    discourse_markers: List[str]
    rhetorical_structure: Dict[str, float]
    
    # Cross-lingual features
    translation_equivalents: Dict[str, List[str]]
    cultural_adaptation_score: float
    language_specific_features: Dict[str, float]
    
    # Statistical features
    entropy: float
    perplexity: float
    information_density: float


class AdvancedLinguisticAnalyzer:
    """
    Advanced linguistic analysis for research-grade semantic analysis.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        additional_models: Optional[Dict[str, str]] = None,
    ):
        """Initialize the advanced linguistic analyzer."""
        self.primary_model = spacy.load(model_name)
        self.models = {model_name.split("_")[0]: self.primary_model}
        
        if additional_models:
            for lang, model in additional_models.items():
                self.models[lang] = spacy.load(model)
        
        # Initialize semantic field mappings
        self.semantic_fields = self._initialize_semantic_fields()
        
        # Initialize rhetorical structure patterns
        self.rhetorical_patterns = self._initialize_rhetorical_patterns()

    def analyze_text(
        self,
        text: str,
        language: str = "en",
        reference_texts: Optional[List[str]] = None,
    ) -> AdvancedLinguisticFeatures:
        """
        Perform advanced linguistic analysis on text.
        
        Args:
            text: Text to analyze
            language: Language code
            reference_texts: Optional reference texts for comparison
            
        Returns:
            AdvancedLinguisticFeatures object
        """
        nlp = self.models.get(language, self.primary_model)
        doc = nlp(text)
        
        # Syntactic analysis
        dependency_tree_depth = self._analyze_dependency_tree(doc)
        clause_complexity = self._analyze_clause_complexity(doc)
        phrase_structure_complexity = self._analyze_phrase_structure(doc)
        
        # Semantic analysis
        lexical_diversity = self._calculate_lexical_diversity(doc)
        semantic_field_density = self._analyze_semantic_fields(doc)
        conceptual_metaphors = self._detect_conceptual_metaphors(doc)
        
        # Discourse analysis
        coherence_score = self._calculate_coherence(doc)
        discourse_markers = self._extract_discourse_markers(doc)
        rhetorical_structure = self._analyze_rhetorical_structure(doc)
        
        # Cross-lingual analysis
        translation_equivalents = self._find_translation_equivalents(doc, language)
        cultural_adaptation_score = self._calculate_cultural_adaptation(doc)
        language_specific_features = self._analyze_language_specific_features(doc)
        
        # Statistical analysis
        entropy = self._calculate_entropy(doc)
        perplexity = self._calculate_perplexity(doc, reference_texts)
        information_density = self._calculate_information_density(doc)
        
        return AdvancedLinguisticFeatures(
            dependency_tree_depth=dependency_tree_depth,
            clause_complexity=clause_complexity,
            phrase_structure_complexity=phrase_structure_complexity,
            lexical_diversity=lexical_diversity,
            semantic_field_density=semantic_field_density,
            conceptual_metaphors=conceptual_metaphors,
            coherence_score=coherence_score,
            discourse_markers=discourse_markers,
            rhetorical_structure=rhetorical_structure,
            translation_equivalents=translation_equivalents,
            cultural_adaptation_score=cultural_adaptation_score,
            language_specific_features=language_specific_features,
            entropy=entropy,
            perplexity=perplexity,
            information_density=information_density,
        )

    def _analyze_dependency_tree(self, doc: Doc) -> float:
        """Analyze the depth and complexity of the dependency tree."""
        # Create a directed graph from the dependency tree
        G = nx.DiGraph()
        
        # Add nodes and edges
        for token in doc:
            G.add_node(token.i, text=token.text)
            if token.head.i != token.i:  # Skip self-loops
                G.add_edge(token.head.i, token.i)
        
        # Calculate tree metrics
        if not G.nodes():
            return 0.0
            
        # Calculate average depth
        depths = []
        for node in G.nodes():
            if G.in_degree(node) == 0:  # Root node
                depth = max(len(p) for p in nx.all_simple_paths(G, node, target=None))
                depths.append(depth)
        
        return np.mean(depths) if depths else 0.0

    def _analyze_clause_complexity(self, doc: Doc) -> float:
        """Analyze the complexity of clause structures."""
        # Count different types of clauses
        clause_types = {
            "main": 0,
            "subordinate": 0,
            "relative": 0,
            "complement": 0,
        }
        
        for token in doc:
            if token.dep_ == "ROOT":
                clause_types["main"] += 1
            elif token.dep_ in {"ccomp", "xcomp"}:
                clause_types["complement"] += 1
            elif token.dep_ == "relcl":
                clause_types["relative"] += 1
            elif token.dep_ in {"advcl", "acl"}:
                clause_types["subordinate"] += 1
        
        # Calculate complexity score
        total_clauses = sum(clause_types.values())
        if total_clauses == 0:
            return 0.0
            
        # Weight different clause types
        weights = {
            "main": 1.0,
            "subordinate": 2.0,
            "relative": 1.5,
            "complement": 1.5,
        }
        
        complexity = sum(
            clause_types[ctype] * weights[ctype]
            for ctype in clause_types
        ) / total_clauses
        
        return complexity

    def _analyze_phrase_structure(self, doc: Doc) -> float:
        """Analyze the complexity of phrase structures."""
        # Count different types of phrases
        phrase_types = {
            "NP": 0,  # Noun phrases
            "VP": 0,  # Verb phrases
            "PP": 0,  # Prepositional phrases
            "AP": 0,  # Adjective phrases
        }
        
        for chunk in doc.noun_chunks:
            phrase_types["NP"] += 1
            
        for token in doc:
            if token.pos_ == "VERB":
                phrase_types["VP"] += 1
            elif token.pos_ == "ADP":
                phrase_types["PP"] += 1
            elif token.pos_ == "ADJ":
                phrase_types["AP"] += 1
        
        # Calculate complexity score
        total_phrases = sum(phrase_types.values())
        if total_phrases == 0:
            return 0.0
            
        # Weight different phrase types
        weights = {
            "NP": 1.0,
            "VP": 1.5,
            "PP": 1.0,
            "AP": 0.5,
        }
        
        complexity = sum(
            phrase_types[ptype] * weights[ptype]
            for ptype in phrase_types
        ) / total_phrases
        
        return complexity

    def _calculate_lexical_diversity(self, doc: Doc) -> float:
        """Calculate lexical diversity using type-token ratio."""
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        if not tokens:
            return 0.0
            
        types = len(set(tokens))
        tokens = len(tokens)
        
        return types / tokens

    def _analyze_semantic_fields(self, doc: Doc) -> Dict[str, float]:
        """Analyze the distribution of semantic fields."""
        field_counts = Counter()
        
        for token in doc:
            if token.pos_ in {"NOUN", "VERB", "ADJ"}:
                # Map token to semantic field
                field = self._get_semantic_field(token)
                field_counts[field] += 1
        
        total = sum(field_counts.values())
        if total == 0:
            return {"general": 1.0}
            
        return {
            field: count / total
            for field, count in field_counts.items()
        }

    def _detect_conceptual_metaphors(self, doc: Doc) -> List[str]:
        """Detect conceptual metaphors in the text."""
        metaphors = []
        
        # Common conceptual metaphor patterns
        metaphor_patterns = {
            "TIME_IS_SPACE": [
                ("ahead", "behind", "forward", "backward"),
                ("time", "space", "journey", "path"),
            ],
            "LIFE_IS_JOURNEY": [
                ("path", "road", "journey", "travel"),
                ("life", "living", "existence", "being"),
            ],
            "ARGUMENT_IS_WAR": [
                ("attack", "defend", "win", "lose"),
                ("argument", "debate", "discussion", "point"),
            ],
        }
        
        # Check for metaphor patterns
        for metaphor, (source, target) in metaphor_patterns.items():
            source_words = set(token.text.lower() for token in doc if token.text.lower() in source)
            target_words = set(token.text.lower() for token in doc if token.text.lower() in target)
            
            if source_words and target_words:
                metaphors.append(metaphor)
        
        return metaphors

    def _calculate_coherence(self, doc: Doc) -> float:
        """Calculate text coherence score."""
        # Calculate sentence similarity matrix
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0
            
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences):
                if i != j:
                    similarity = self._calculate_sentence_similarity(sent1, sent2)
                    similarity_matrix[i, j] = similarity
        
        # Calculate coherence as average similarity
        return np.mean(similarity_matrix)

    def _calculate_sentence_similarity(self, sent1: Doc, sent2: Doc) -> float:
        """Calculate similarity between two sentences."""
        # Get sentence embeddings
        vec1 = sent1.vector
        vec2 = sent2.vector
        
        # Calculate cosine similarity
        return 1 - cosine(vec1, vec2)

    def _extract_discourse_markers(self, doc: Doc) -> List[str]:
        """Extract discourse markers from text."""
        discourse_markers = {
            "addition": {"furthermore", "moreover", "additionally"},
            "contrast": {"however", "nevertheless", "conversely"},
            "cause": {"because", "since", "therefore"},
            "time": {"meanwhile", "subsequently", "previously"},
            "emphasis": {"indeed", "certainly", "undoubtedly"},
        }
        
        markers = []
        for token in doc:
            for category, markers_set in discourse_markers.items():
                if token.text.lower() in markers_set:
                    markers.append(f"{token.text} ({category})")
        
        return markers

    def _analyze_rhetorical_structure(self, doc: Doc) -> Dict[str, float]:
        """Analyze rhetorical structure of the text."""
        structure_scores = {
            "narrative": 0.0,
            "argumentative": 0.0,
            "descriptive": 0.0,
            "expository": 0.0,
        }
        
        # Count rhetorical markers
        for pattern, markers in self.rhetorical_patterns.items():
            count = sum(
                1 for token in doc
                if token.text.lower() in markers
            )
            structure_scores[pattern] = count / len(doc)
        
        return structure_scores

    def _find_translation_equivalents(
        self,
        doc: Doc,
        source_language: str,
    ) -> Dict[str, List[str]]:
        """Find translation equivalents in other languages."""
        # This would typically involve a translation model or dictionary
        return {}  # Placeholder

    def _calculate_cultural_adaptation(self, doc: Doc) -> float:
        """Calculate cultural adaptation score."""
        # This would typically involve cultural knowledge bases
        return 0.0  # Placeholder

    def _analyze_language_specific_features(self, doc: Doc) -> Dict[str, float]:
        """Analyze language-specific features."""
        features = {
            "morphological_complexity": 0.0,
            "syntactic_flexibility": 0.0,
            "lexical_richness": 0.0,
        }
        
        # Calculate morphological complexity
        morphemes = sum(len(token.morph) for token in doc)
        words = len(doc)
        features["morphological_complexity"] = morphemes / words if words > 0 else 0
        
        # Calculate syntactic flexibility
        pos_tags = Counter(token.pos_ for token in doc)
        total_tags = sum(pos_tags.values())
        features["syntactic_flexibility"] = len(pos_tags) / total_tags if total_tags > 0 else 0
        
        # Calculate lexical richness
        unique_words = len(set(token.text.lower() for token in doc if token.is_alpha))
        total_words = len([token for token in doc if token.is_alpha])
        features["lexical_richness"] = unique_words / total_words if total_words > 0 else 0
        
        return features

    def _calculate_entropy(self, doc: Doc) -> float:
        """Calculate the entropy of the text."""
        # Get word frequencies
        word_freq = Counter(token.text.lower() for token in doc if token.is_alpha)
        total_words = sum(word_freq.values())
        
        if total_words == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in word_freq.values():
            p = count / total_words
            entropy -= p * math.log2(p)
        
        return entropy

    def _calculate_perplexity(
        self,
        doc: Doc,
        reference_texts: Optional[List[str]] = None,
    ) -> float:
        """Calculate perplexity of the text."""
        if not reference_texts:
            return 0.0
            
        # Create language model from reference texts
        vectorizer = TfidfVectorizer()
        try:
            vectorizer.fit(reference_texts)
            doc_vector = vectorizer.transform([doc.text])
            ref_vectors = vectorizer.transform(reference_texts)
            
            # Calculate average cosine distance
            distances = [
                cosine(doc_vector.toarray()[0], ref_vec.toarray()[0])
                for ref_vec in ref_vectors
            ]
            
            return np.mean(distances)
        except:
            return 0.0

    def _calculate_information_density(self, doc: Doc) -> float:
        """Calculate information density of the text."""
        # Consider factors like:
        # - Content words vs. function words
        # - Unique concepts
        # - Semantic relationships
        
        content_words = len([token for token in doc if not token.is_stop])
        total_words = len(doc)
        
        if total_words == 0:
            return 0.0
            
        # Calculate basic density
        density = content_words / total_words
        
        # Adjust for semantic relationships
        semantic_relations = len([
            token for token in doc
            if token.dep_ in {"nsubj", "dobj", "iobj", "pobj"}
        ])
        
        density *= (1 + semantic_relations / total_words)
        
        return min(density, 1.0)

    def _initialize_semantic_fields(self) -> Dict[str, Set[str]]:
        """Initialize semantic field mappings."""
        return {
            "abstract": {"idea", "concept", "thought", "theory"},
            "concrete": {"object", "thing", "item", "entity"},
            "action": {"do", "make", "create", "perform"},
            "state": {"be", "exist", "remain", "stay"},
            "property": {"have", "possess", "own", "contain"},
            "time": {"time", "moment", "period", "duration"},
            "space": {"space", "place", "location", "position"},
        }

    def _initialize_rhetorical_patterns(self) -> Dict[str, Set[str]]:
        """Initialize rhetorical structure patterns."""
        return {
            "narrative": {
                "then", "after", "before", "while", "when",
                "suddenly", "finally", "eventually",
            },
            "argumentative": {
                "because", "therefore", "thus", "hence",
                "however", "nevertheless", "although",
            },
            "descriptive": {
                "like", "such as", "for example",
                "appears", "seems", "looks",
            },
            "expository": {
                "first", "second", "finally",
                "in addition", "moreover", "furthermore",
            },
        }

    def _get_semantic_field(self, token: Token) -> str:
        """Get the semantic field of a token."""
        for field, words in self.semantic_fields.items():
            if token.text.lower() in words:
                return field
        return "general" 