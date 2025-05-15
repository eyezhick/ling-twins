"""
Linguistic analysis functionality for semantic doppelgängers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc


@dataclass
class LinguisticFeatures:
    """Represents linguistic features of a text."""
    style: str  # formal, informal, poetic, etc.
    register: str  # academic, conversational, literary, etc.
    cultural_context: str  # cultural references and context
    syntactic_complexity: float  # 0-1 score of syntactic complexity
    semantic_density: float  # 0-1 score of semantic information density
    temporal_markers: List[str]  # temporal expressions
    cultural_markers: List[str]  # culture-specific references
    pragmatic_markers: List[str]  # pragmatic markers (e.g., "actually", "well")


class LinguisticAnalyzer:
    """
    Analyzes linguistic features of texts for semantic doppelgänger detection.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        additional_models: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the linguistic analyzer.

        Args:
            model_name: Name of the primary spaCy model
            additional_models: Optional dict of language code to model name
        """
        self.primary_model = spacy.load(model_name)
        self.models = {model_name.split("_")[0]: self.primary_model}
        
        if additional_models:
            for lang, model in additional_models.items():
                self.models[lang] = spacy.load(model)

    def analyze_text(
        self,
        text: str,
        language: str = "en",
    ) -> LinguisticFeatures:
        """
        Analyze linguistic features of a text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            LinguisticFeatures object
        """
        # Get appropriate model
        nlp = self.models.get(language, self.primary_model)
        doc = nlp(text)

        # Extract features
        style = self._detect_style(doc)
        register = self._detect_register(doc)
        cultural_context = self._detect_cultural_context(doc)
        syntactic_complexity = self._calculate_syntactic_complexity(doc)
        semantic_density = self._calculate_semantic_density(doc)
        temporal_markers = self._extract_temporal_markers(doc)
        cultural_markers = self._extract_cultural_markers(doc)
        pragmatic_markers = self._extract_pragmatic_markers(doc)

        return LinguisticFeatures(
            style=style,
            register=register,
            cultural_context=cultural_context,
            syntactic_complexity=syntactic_complexity,
            semantic_density=semantic_density,
            temporal_markers=temporal_markers,
            cultural_markers=cultural_markers,
            pragmatic_markers=pragmatic_markers,
        )

    def _detect_style(self, doc: Doc) -> str:
        """Detect the style of a text."""
        # Analyze sentence structure
        avg_sentence_length = len(doc) / len(list(doc.sents))
        
        # Check for poetic features
        has_rhyme = self._check_rhyme(doc)
        has_alliteration = self._check_alliteration(doc)
        
        # Check for formal features
        has_formal_markers = any(token.text.lower() in {
            "therefore", "furthermore", "moreover", "consequently"
        } for token in doc)
        
        if has_rhyme or has_alliteration:
            return "poetic"
        elif has_formal_markers and avg_sentence_length > 20:
            return "formal"
        elif avg_sentence_length < 10:
            return "conversational"
        else:
            return "neutral"

    def _detect_register(self, doc: Doc) -> str:
        """Detect the register of a text."""
        # Check for academic markers
        academic_markers = {
            "research", "study", "analysis", "conclude", "demonstrate",
            "evidence", "theory", "hypothesis", "methodology"
        }
        
        # Check for literary markers
        literary_markers = {
            "metaphor", "simile", "imagery", "symbol", "theme",
            "narrative", "character", "plot", "setting"
        }
        
        # Check for technical markers
        technical_markers = {
            "function", "parameter", "algorithm", "protocol", "interface",
            "system", "process", "implementation", "architecture"
        }
        
        text_lower = doc.text.lower()
        
        if any(marker in text_lower for marker in academic_markers):
            return "academic"
        elif any(marker in text_lower for marker in literary_markers):
            return "literary"
        elif any(marker in text_lower for marker in technical_markers):
            return "technical"
        else:
            return "general"

    def _detect_cultural_context(self, doc: Doc) -> str:
        """Detect cultural context and references."""
        # This would typically involve a more sophisticated analysis
        # using cultural knowledge bases or reference databases
        return "general"  # Placeholder

    def _calculate_syntactic_complexity(self, doc: Doc) -> float:
        """Calculate syntactic complexity score."""
        # Consider factors like:
        # - Average sentence length
        # - Number of subordinate clauses
        # - Depth of parse tree
        # - Use of complex constructions
        
        avg_sentence_length = len(doc) / len(list(doc.sents))
        subclauses = len([token for token in doc if token.dep_ in {
            "ccomp", "xcomp", "advcl", "relcl"
        }])
        
        # Normalize to 0-1 range
        complexity = (avg_sentence_length / 50 + subclauses / len(doc)) / 2
        return min(max(complexity, 0), 1)

    def _calculate_semantic_density(self, doc: Doc) -> float:
        """Calculate semantic information density."""
        # Consider factors like:
        # - Number of content words vs. function words
        # - Lexical diversity
        # - Information-theoretic measures
        
        content_words = len([token for token in doc if not token.is_stop])
        total_words = len(doc)
        
        if total_words == 0:
            return 0
            
        return content_words / total_words

    def _extract_temporal_markers(self, doc: Doc) -> List[str]:
        """Extract temporal expressions."""
        return [
            ent.text for ent in doc.ents
            if ent.label_ in {"DATE", "TIME"}
        ]

    def _extract_cultural_markers(self, doc: Doc) -> List[str]:
        """Extract culture-specific references."""
        # This would typically involve matching against a cultural reference database
        return []  # Placeholder

    def _extract_pragmatic_markers(self, doc: Doc) -> List[str]:
        """Extract pragmatic markers."""
        pragmatic_markers = {
            "actually", "well", "you know", "I mean", "like",
            "basically", "literally", "honestly", "frankly"
        }
        
        return [
            token.text for token in doc
            if token.text.lower() in pragmatic_markers
        ]

    def _check_rhyme(self, doc: Doc) -> bool:
        """Check for rhyming patterns."""
        # This would typically involve more sophisticated rhyme detection
        return False  # Placeholder

    def _check_alliteration(self, doc: Doc) -> bool:
        """Check for alliteration patterns."""
        words = [token.text.lower() for token in doc if token.is_alpha]
        if len(words) < 2:
            return False
            
        # Check for consecutive words starting with the same letter
        for i in range(len(words) - 1):
            if words[i][0] == words[i + 1][0]:
                return True
                
        return False 