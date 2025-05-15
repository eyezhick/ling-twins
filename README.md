# Semantic Doppelgängers 🌍

*"Language is the dress of thought." - Samuel Johnson*

## The Story Behind This Project

Have you ever marveled at how the same profound truth can be expressed in wildly different ways across languages and cultures? Consider how "The early bird catches the worm" in English becomes "A quien madruga, Dios le ayuda" (God helps those who rise early) in Spanish, or how Shakespeare's "All the world's a stage" finds its echo in modern tweets about life being like a video game.

This project was born from a fascination with these linguistic parallels - the way different cultures, time periods, and contexts express the same semantic core through unique linguistic structures. It's not just about translation or simple paraphrasing; it's about uncovering the hidden semantic DNA that connects seemingly disparate expressions across the linguistic landscape.

## What We're Building

Semantic Doppelgängers is a sophisticated tool that bridges the gap between computational linguistics and cross-cultural language analysis. It employs state-of-the-art multilingual embeddings to discover semantic parallels that traditional NLP approaches might miss.

### Key Features

- 🔍 Cross-lingual semantic search with support for 100+ languages
- 📚 Multi-domain analysis (literature, social media, news, poetry)
- 🌐 Cultural and temporal language variant detection
- 🎯 Stylistic and syntactic variation analysis
- 🧠 Advanced embedding space alignment techniques
- 📊 Linguistic feature extraction and comparison
- 🔄 Bidirectional semantic mapping across languages
- 🎨 Style transfer and cultural adaptation

### Technical Architecture

#### Core Components

1. **Multilingual Embedding System**
   - State-of-the-art transformer-based models (paraphrase-multilingual-mpnet-base-v2)
   - FAISS-based efficient similarity search
   - Cross-lingual embedding space alignment

2. **Advanced Linguistic Analysis**
   - Dependency tree analysis using NetworkX
   - Clause and phrase structure complexity metrics
   - Rhetorical structure analysis
   - Discourse coherence scoring
   - Conceptual metaphor detection

3. **Statistical Analysis**
   - Information-theoretic measures (entropy, perplexity)
   - Lexical diversity metrics
   - Semantic field distribution analysis
   - Language-specific feature extraction

#### Mathematical Foundations

1. **Similarity Metrics**
   - Cosine similarity for semantic matching
   - L2 distance for embedding space navigation
   - Custom weighted scoring for multi-feature comparison

2. **Complexity Measures**
   - Syntactic complexity scoring
   - Information density calculation
   - Morphological complexity analysis
   - Lexical diversity metrics

3. **Statistical Analysis**
   - Entropy-based information content analysis
   - Perplexity scoring for language model evaluation
   - TF-IDF based feature extraction
   - Distribution analysis across semantic fields

### Linguistic Capabilities

#### Syntactic Analysis
- Dependency tree depth and complexity
- Clause structure analysis
- Phrase structure complexity
- Syntactic flexibility scoring
- Morphological complexity analysis

#### Semantic Analysis
- Lexical diversity measurement
- Semantic field distribution
- Conceptual metaphor detection
- Information density calculation
- Cross-lingual semantic mapping

#### Discourse Analysis
- Coherence scoring
- Rhetorical structure analysis
- Discourse marker detection
- Text structure classification
- Cross-sentence relationship analysis

#### Cross-Cultural Analysis
- Cultural adaptation scoring
- Language-specific feature extraction
- Translation equivalent detection
- Cultural marker identification
- Cross-lingual pattern matching

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-doppelgangers.git
cd semantic-doppelgangers

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from semantic_doppelgangers import DoppelgangerFinder

# Initialize the finder with linguistic analysis enabled
finder = DoppelgangerFinder(
    model_name="paraphrase-multilingual-mpnet-base-v2",
    enable_linguistic_features=True,
    enable_advanced_analysis=True
)

# Find semantic twins with linguistic analysis
results = finder.find_doppelgangers(
    "Life is like a box of chocolates",
    top_k=5,
    domains=["literature", "social_media"],
    analyze_linguistic_features=True,
    analyze_advanced_features=True
)

# Print results with linguistic analysis
for result in results:
    print(f"Similarity: {result.similarity:.2f}")
    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Source: {result.source}")
    
    if result.linguistic_features:
        print("\nLinguistic Features:")
        print(f"- Style: {result.linguistic_features.style}")
        print(f"- Register: {result.linguistic_features.register}")
        print(f"- Syntactic Complexity: {result.linguistic_features.syntactic_complexity:.2f}")
        print(f"- Semantic Density: {result.linguistic_features.semantic_density:.2f}")
    
    if result.advanced_features:
        print("\nAdvanced Analysis:")
        print(f"- Dependency Tree Depth: {result.advanced_features.dependency_tree_depth:.2f}")
        print(f"- Clause Complexity: {result.advanced_features.clause_complexity:.2f}")
        print(f"- Lexical Diversity: {result.advanced_features.lexical_diversity:.2f}")
        print(f"- Coherence Score: {result.advanced_features.coherence_score:.2f}")
        print(f"- Information Density: {result.advanced_features.information_density:.2f}")
        print(f"- Conceptual Metaphors: {result.advanced_features.conceptual_metaphors}")
    print("---")
```

## Project Structure

```
semantic_doppelgangers/
├── data/                  # Data processing and storage
├── models/               # Model architectures and weights
├── training/            # Training scripts and utilities
├── evaluation/          # Evaluation metrics and benchmarks
├── linguistic/          # Linguistic analysis modules
│   ├── features/       # Feature extraction
│   ├── analysis/       # Linguistic analysis
│   └── visualization/  # Result visualization
└── examples/            # Example notebooks and use cases
```

## Research Applications

### Computational Linguistics
- Study semantic universals across languages
- Analyze syntactic patterns in cross-lingual contexts
- Investigate information structure variations
- Research discourse patterns across cultures

### Cultural Anthropology
- Analyze cross-cultural expression patterns
- Study cultural adaptation in language
- Investigate conceptual metaphor universals
- Research rhetorical structure variations

### Natural Language Processing
- Improve cross-lingual understanding
- Enhance semantic preservation in translation
- Develop better multilingual models
- Create more culturally aware NLP systems

### Language Education
- Aid in understanding cultural context
- Provide insights into language structure
- Help identify cross-lingual patterns
- Support cultural adaptation learning

## Technical Details

### Dependencies
- PyTorch for deep learning
- Transformers for model architecture
- FAISS for efficient similarity search
- NetworkX for graph analysis
- spaCy for linguistic processing
- SciPy for scientific computing
- scikit-learn for machine learning
- Matplotlib/Seaborn/Plotly for visualization

### Performance Considerations
- GPU acceleration for embedding computation
- Efficient indexing with FAISS
- Batch processing for large corpora
- Caching for frequent analyses
- Memory-efficient data structures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*"The limits of my language mean the limits of my world." - Ludwig Wittgenstein*

### Technical Architecture

#### Core Components

1. **Multilingual Embedding System**
   ```python
   # Example of embedding space alignment
   def align_embeddings(source_emb, target_emb):
       # Procrustes alignment
       U, _, Vt = np.linalg.svd(target_emb.T @ source_emb)
       W = U @ Vt
       return source_emb @ W
   ```

2. **Advanced Linguistic Analysis**
   ```python
   # Dependency tree analysis
   def analyze_dependency_tree(doc):
       G = nx.DiGraph()
       for token in doc:
           G.add_node(token.i, text=token.text)
           if token.head.i != token.i:
               G.add_edge(token.head.i, token.i)
       return G
   ```

#### Mathematical Foundations

1. **Similarity Metrics**
   - Cosine Similarity:
     ```
     sim(x,y) = (x·y)/(||x||·||y||)
     ```
   - L2 Distance:
     ```
     d(x,y) = √(Σ(x_i - y_i)²)
     ```
   - Weighted Multi-feature Score:
     ```
     score = Σ(w_i * f_i) / Σ(w_i)
     ```

2. **Complexity Measures**
   - Syntactic Complexity:
     ```
     C_syn = (w_main + 2w_sub + 1.5w_rel + 1.5w_comp) / N_clauses
     ```
   - Information Density:
     ```
     D_info = (N_content + N_relations) / N_total
     ```
   - Lexical Diversity (Type-Token Ratio):
     ```
     TTR = N_types / N_tokens
     ```

3. **Statistical Analysis**
   - Entropy:
     ```
     H = -Σ(p_i * log2(p_i))
     ```
   - Perplexity:
     ```
     PP = 2^(-1/N * Σ(log2(P(x_i))))
     ```

### Implementation Examples

#### 1. Cross-Lingual Semantic Analysis
```python
# Example of finding semantic equivalents
text_en = "The early bird catches the worm"
text_es = "A quien madruga, Dios le ayuda"

# Get embeddings
emb_en = model.encode([text_en])[0]
emb_es = model.encode([text_es])[0]

# Calculate similarity
similarity = 1 - cosine(emb_en, emb_es)
print(f"Semantic similarity: {similarity:.3f}")

# Analyze linguistic features
features_en = analyzer.analyze_text(text_en)
features_es = analyzer.analyze_text(text_es)

# Compare features
print("\nFeature Comparison:")
print(f"EN - Syntactic Complexity: {features_en.clause_complexity:.2f}")
print(f"ES - Syntactic Complexity: {features_es.clause_complexity:.2f}")
print(f"EN - Lexical Diversity: {features_en.lexical_diversity:.2f}")
print(f"ES - Lexical Diversity: {features_es.lexical_diversity:.2f}")
```

#### 2. Dependency Tree Analysis
```python
# Example of dependency tree analysis
text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# Create and analyze dependency tree
G = analyze_dependency_tree(doc)

# Calculate tree metrics
depth = max(len(p) for p in nx.all_simple_paths(G, 0, target=None))
complexity = len(list(nx.descendants(G, 0)))

print(f"Tree Depth: {depth}")
print(f"Tree Complexity: {complexity}")

# Visualize tree
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=10)
```

#### 3. Rhetorical Structure Analysis
```python
# Example of rhetorical structure analysis
text = """
First, we must consider the environmental impact. 
Furthermore, economic factors play a crucial role. 
However, social implications cannot be ignored.
"""

# Analyze rhetorical structure
structure = analyzer.analyze_rhetorical_structure(text)

# Plot structure distribution
plt.figure(figsize=(8, 4))
plt.bar(structure.keys(), structure.values())
plt.title('Rhetorical Structure Distribution')
plt.xticks(rotation=45)
```

### Research Applications

#### 1. Computational Linguistics
```python
# Example of cross-lingual pattern analysis
def analyze_cross_lingual_patterns(texts_en, texts_es):
    patterns = {
        'metaphor': [],
        'idiom': [],
        'proverb': []
    }
    
    for text_en, text_es in zip(texts_en, texts_es):
        # Analyze conceptual metaphors
        metaphors_en = analyzer.detect_conceptual_metaphors(text_en)
        metaphors_es = analyzer.detect_conceptual_metaphors(text_es)
        
        # Compare patterns
        if metaphors_en and metaphors_es:
            patterns['metaphor'].append({
                'en': metaphors_en,
                'es': metaphors_es,
                'similarity': calculate_metaphor_similarity(metaphors_en, metaphors_es)
            })
    
    return patterns
```

#### 2. Cultural Anthropology
```python
# Example of cultural adaptation analysis
def analyze_cultural_adaptation(text, source_culture, target_culture):
    # Extract cultural markers
    markers = analyzer.extract_cultural_markers(text)
    
    # Calculate adaptation score
    adaptation_score = 0
    for marker in markers:
        if marker in cultural_knowledge_base[target_culture]:
            adaptation_score += 1
    
    return {
        'original_markers': markers,
        'adaptation_score': adaptation_score / len(markers),
        'cultural_context': analyzer.detect_cultural_context(text)
    }
```

### Performance Analysis

#### 1. Embedding Space Visualization
```python
# Example of embedding space visualization
def visualize_embedding_space(texts, labels):
    # Get embeddings
    embeddings = model.encode(texts)
    
    # Reduce dimensionality
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.title('Embedding Space Distribution')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
```

#### 2. Performance Metrics
```python
# Example of performance evaluation
def evaluate_performance(test_cases):
    metrics = {
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for case in test_cases:
        results = finder.find_doppelgangers(
            case['query'],
            k=5,
            analyze_linguistic_features=True
        )
        
        # Calculate metrics
        precision = calculate_precision(results, case['relevant'])
        recall = calculate_recall(results, case['relevant'])
        f1 = 2 * (precision * recall) / (precision + recall)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```
