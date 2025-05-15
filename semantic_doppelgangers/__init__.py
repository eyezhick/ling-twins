"""
Semantic Doppelgängers: Finding semantic twins across languages and domains.
"""

__version__ = "0.1.0"

from semantic_doppelgangers.finder import DoppelgangerFinder
from semantic_doppelgangers.models import MultilingualEmbedder
from semantic_doppelgangers.data import CorpusLoader

__all__ = ["DoppelgangerFinder", "MultilingualEmbedder", "CorpusLoader"] 