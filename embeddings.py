import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Dict, List
import torch
import numpy as np

class HashNGramEmbedding(eqx.Module):
    """Hash n-gram Embeddings."""
    hash_sizes: List[int]
    hash_table_size: int
    n_embd: int
    hash_embeddings: Dict[str, jnp.ndarray]

    def __init__(self, hash_sizes: List[int], hash_table_size: int, n_embd: int, *, key):
        super().__init__()
        self.hash_sizes = hash_sizes
        self.hash_table_size = hash_table_size
        self.n_embd = n_embd
        
        # Create separate keys for each embedding
        keys = jax.random.split(key, len(hash_sizes))
        self.hash_embeddings = {
            f"hash_{n}": jax.random.normal(k, (hash_table_size, n_embd)) 
            for k, n in zip(keys, hash_sizes)
        }

    def __call__(self, x):
        B, T = x.shape
        embeddings = jnp.zeros((B, T, self.n_embd))
        
        for n in self.hash_sizes:
            if T < n:
                continue
            # Extract n-grams using sliding window
            ngrams = jnp.stack([x[:, i:i+n] for i in range(T-n+1)], axis=1)
            # Compute hash
            hashes = self.roll_poly_hash(ngrams)
            hashes = hashes % self.hash_table_size
            # Lookup embeddings
            hash_emb = jnp.take(self.hash_embeddings[f"hash_{n}"], hashes, axis=0)
            # Update embeddings
            embeddings = embeddings.at[:, n-1:T].add(hash_emb)
            
        # Normalize
        embeddings = embeddings / len(self.hash_sizes)
        return embeddings

    def roll_poly_hash(self, ngrams):
        """Simple polynomial rolling hash."""
        base = 257
        hash_val = jnp.zeros((ngrams.shape[0], ngrams.shape[1]), dtype=jnp.int32)
        
        for i in range(ngrams.shape[2]):
            # Use smaller intermediate values and more frequent modulo
            hash_val = (hash_val * base) % self.hash_table_size
            hash_val = (hash_val + ngrams[:, :, i].astype(jnp.int32)) % self.hash_table_size
        return hash_val


