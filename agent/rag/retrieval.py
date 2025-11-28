"""TF-IDF based document retrieval with chunking."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    """A document chunk with metadata."""
    id: str
    content: str
    source: str  # filename without extension
    chunk_index: int
    score: float = 0.0
    
    def __repr__(self) -> str:
        return f"Chunk({self.id}, score={self.score:.3f})"


class DocumentRetriever:
    """TF-IDF based retriever over document chunks."""
    
    def __init__(self, docs_dir: str | Path, chunk_size: int = 200):
        """
        Initialize the retriever.
        
        Args:
            docs_dir: Directory containing markdown documents
            chunk_size: Approximate max tokens per chunk
        """
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunks: list[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        self._load_and_chunk_documents()
        self._build_index()
    
    def _load_and_chunk_documents(self) -> None:
        """Load all markdown documents and create chunks."""
        self.chunks = []
        
        for doc_path in sorted(self.docs_dir.glob("*.md")):
            source = doc_path.stem  # filename without extension
            content = doc_path.read_text(encoding="utf-8")
            
            # Split into chunks by sections/paragraphs
            doc_chunks = self._chunk_document(content, source)
            self.chunks.extend(doc_chunks)
    
    def _chunk_document(self, content: str, source: str) -> list[Chunk]:
        """Split a document into chunks."""
        chunks = []
        
        # Split by markdown headers or double newlines
        sections = re.split(r'\n(?=#+\s)|(?:\n\s*\n)', content)
        
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If adding this section exceeds chunk size, save current and start new
            if len(current_chunk.split()) + len(section.split()) > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        id=f"{source}::chunk{chunk_index}",
                        content=current_chunk.strip(),
                        source=source,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                current_chunk = section
            else:
                current_chunk = f"{current_chunk}\n{section}".strip() if current_chunk else section
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                id=f"{source}::chunk{chunk_index}",
                content=current_chunk.strip(),
                source=source,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _build_index(self) -> None:
        """Build the TF-IDF index."""
        if not self.chunks:
            return
            
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000
        )
        
        # Build the matrix
        texts = [chunk.content for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of chunks to return
            
        Returns:
            List of Chunk objects with scores
        """
        if not self.chunks or self.vectorizer is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create result chunks with scores
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk_copy = Chunk(
                id=chunk.id,
                content=chunk.content,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=float(similarities[idx])
            )
            results.append(chunk_copy)
        
        return results
    
    def retrieve_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific source document."""
        return [c for c in self.chunks if c.source == source]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_all_chunks(self) -> list[Chunk]:
        """Get all chunks."""
        return self.chunks.copy()
    
    def search_keyword(self, keyword: str) -> list[Chunk]:
        """Simple keyword search (case-insensitive)."""
        keyword_lower = keyword.lower()
        results = []
        for chunk in self.chunks:
            if keyword_lower in chunk.content.lower():
                results.append(chunk)
        return results
    
    def get_sources(self) -> list[str]:
        """Get list of unique source documents."""
        return list(set(chunk.source for chunk in self.chunks))
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __repr__(self) -> str:
        return f"DocumentRetriever({len(self.chunks)} chunks from {len(self.get_sources())} sources)"


class HybridRetriever:
    """Combines TF-IDF with keyword matching for better recall."""
    
    def __init__(self, docs_dir: str | Path, chunk_size: int = 200):
        self.tfidf_retriever = DocumentRetriever(docs_dir, chunk_size)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Retrieve using both TF-IDF and keyword matching.
        
        Combines results and re-ranks by combined score.
        """
        # TF-IDF results
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k=top_k)
        
        # Extract important keywords from query
        keywords = self._extract_keywords(query)
        
        # Keyword results
        keyword_chunks = set()
        for kw in keywords:
            for chunk in self.tfidf_retriever.search_keyword(kw):
                keyword_chunks.add(chunk.id)
        
        # Boost TF-IDF results that also match keywords
        for chunk in tfidf_results:
            if chunk.id in keyword_chunks:
                chunk.score *= 1.2  # 20% boost
        
        # Add keyword-only matches with lower score
        seen_ids = {c.id for c in tfidf_results}
        for chunk_id in keyword_chunks:
            if chunk_id not in seen_ids:
                chunk = self.tfidf_retriever.get_chunk_by_id(chunk_id)
                if chunk:
                    chunk.score = 0.3  # Base score for keyword-only matches
                    tfidf_results.append(chunk)
        
        # Sort by score and return top_k
        tfidf_results.sort(key=lambda x: x.score, reverse=True)
        return tfidf_results[:top_k]
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Simple extraction - look for specific terms
        keywords = []
        
        # Look for quoted terms
        quoted = re.findall(r"'([^']+)'", query)
        keywords.extend(quoted)
        
        # Look for capitalized terms (likely proper nouns/categories)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        keywords.extend(caps)
        
        # Look for specific domain terms
        domain_terms = [
            "AOV", "Average Order Value", "Gross Margin", "KPI",
            "Beverages", "Condiments", "Confections", "Dairy",
            "Summer", "Winter", "return", "policy", "calendar"
        ]
        for term in domain_terms:
            if term.lower() in query.lower():
                keywords.append(term)
        
        return list(set(keywords))
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self.tfidf_retriever.get_chunk_by_id(chunk_id)
    
    def __len__(self) -> int:
        return len(self.tfidf_retriever)
