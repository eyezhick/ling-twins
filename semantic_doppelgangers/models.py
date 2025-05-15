"""
Model architecture and embedding functionality for semantic doppelgängers.
"""

from typing import List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class MultilingualEmbedder:
    """
    Handles multilingual text embedding using transformer models.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder with a multilingual model.

        Args:
            model_name: Name of the model to use
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception:
            # Fallback to manual model loading if SentenceTransformer fails
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.use_sentence_transformer = False
        else:
            self.use_sentence_transformer = True

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.use_sentence_transformer:
            return self.model.get_sentence_embedding_dimension()
        return self.model.config.hidden_size

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize the embeddings

        Returns:
            Tensor of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.use_sentence_transformer:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_tensor=True,
            )
        else:
            # Manual encoding using transformers
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    if normalize:
                        batch_embeddings = torch.nn.functional.normalize(
                            batch_embeddings, p=2, dim=1
                        )
                    
                    embeddings.append(batch_embeddings)
            
            embeddings = torch.cat(embeddings, dim=0)

        return embeddings.cpu().numpy()

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """
        Fine-tune the model on a specific task.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        if not self.use_sentence_transformer:
            raise NotImplementedError(
                "Fine-tuning is only supported with SentenceTransformer models"
            )

        # Convert to sentence-transformers format
        train_examples = list(zip(train_texts, train_labels))
        if val_texts and val_labels:
            val_examples = list(zip(val_texts, val_labels))
        else:
            val_examples = None

        # Fine-tune using sentence-transformers
        self.model.fit(
            train_objectives=[(train_examples, torch.nn.CrossEntropyLoss())],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_data=val_examples,
            show_progress_bar=True,
        ) 