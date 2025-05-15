"""
Example script demonstrating how to use the Semantic Doppelgängers package.
"""

import argparse
from pathlib import Path

from semantic_doppelgangers import CorpusLoader, DoppelgangerFinder


def main():
    parser = argparse.ArgumentParser(description="Find semantic doppelgängers")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--min_similarity", type=float, default=0.7, help="Minimum similarity threshold")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache datasets")
    args = parser.parse_args()

    # Create directories
    Path(args.data_dir).mkdir(exist_ok=True)
    Path(args.cache_dir).mkdir(exist_ok=True)

    # Initialize corpus loader
    loader = CorpusLoader(cache_dir=args.cache_dir)

    # Load some example data
    print("Loading Wikipedia data...")
    wiki_texts, wiki_langs, wiki_sources = loader.load_wikipedia(
        languages=["en", "es", "fr"],
        max_samples=1000,
    )

    print("Loading OpenSubtitles data...")
    sub_texts, sub_langs, sub_sources = loader.load_opensubtitles(
        languages=["en", "es", "fr"],
        max_samples=1000,
    )

    # Combine and preprocess data
    all_texts = wiki_texts + sub_texts
    all_langs = wiki_langs + sub_langs
    all_sources = wiki_sources + sub_sources
    all_domains = ["wikipedia"] * len(wiki_texts) + ["opensubtitles"] * len(sub_texts)

    print("Preprocessing data...")
    texts, langs, sources, domains = loader.preprocess_corpus(
        all_texts,
        all_langs,
        all_sources,
        all_domains,
    )

    # Initialize doppelgänger finder
    finder = DoppelgangerFinder()

    # Add corpus to index
    print("Building search index...")
    finder.add_corpus(texts, sources, langs, domains)

    # Save index for future use
    finder.save_index(args.data_dir)

    # Find doppelgängers
    print(f"\nFinding doppelgängers for: {args.query}")
    results = finder.find_doppelgangers(
        args.query,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    # Print results
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result.similarity:.3f}")
        print(f"Text: {result.text}")
        print(f"Language: {result.language}")
        print(f"Source: {result.source}")
        print(f"Domain: {result.domain}")


if __name__ == "__main__":
    main() 