"""  
Simple knowledge base tool implementation with embedding-based retrieval.
"""
import dataclasses
import numpy
from pathlib import Path
from typing import Callable


# defaults:
ROOT = 'file:///'           # default root URL for documents
DEFAULT_CHUNK_SIZE = 1028   # default max characters per chunk
DEFAULT_CHUNK_OVERLAP = 64  # default overlap characters between chunks


def _normalize_destination(destination: str) -> str:
    """
    Normalize destination path to ensure it ends with a single slash for URL concatenation.
    If destination is empty or just slashes, defaults to ROOT.
    """
    if not destination or destination in ('/', ''):
        return ROOT
    if not destination.startswith('file://'):
        destination = ROOT + destination.lstrip('/')
    if not destination.endswith('/'):
        destination += '/'
    return destination


def cosine_distance(v1: numpy.ndarray, v2: numpy.ndarray, normalized: bool = True) -> float:
    """
    Calculates the cosine distance between two 1D numpy vectors.
    """
    dot_product = numpy.dot(v1, v2)
    if normalized:
        norm_v1 = numpy.linalg.norm(v1)
        norm_v2 = numpy.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0  # Cosine distance is 1 (vectors are orthogonal) if one is a zero vector

        cosine_similarity = dot_product / (norm_v1 * norm_v2)
    else:
        cosine_similarity = dot_product

    return 1 - cosine_similarity


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks by paragraphs, merging short paragraphs up to chunk_size characters.
    Optionally prefix each chunk after the first with the trailing overlap from the previous chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = text.split('\n\n')
    chunks = []
    current = ''
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            current = chunks[-1][-overlap:] + para if overlap else para
        else:
            current = current + '\n\n' + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


@dataclasses.dataclass(slots=True)
class Content:
    """
    Standardized content format for knowledge base entries.
    """
    content: str
    embedding: numpy.ndarray = None  # vector embedding for the content, to be filled in by the knowledge base when added

    # common metadata fields:
    document_url: str | None = None  # original source URL for the content
    chunk_index: int | None = None   # index of this chunk within the original document

    # custom metadata fields:
    metadata: dict = None


@dataclasses.dataclass
class Document:
    """
    Single document with associated chunks and metadata.
    """
    document_url: str
    content: list[Content] = dataclasses.field(default_factory=list)
    metadata: dict = dataclasses.field(default_factory=dict)


class KnowledgeBase:
    """
    A simple tool that allows the agent to store and retrieve information in a key-value store.
    Uses specified async embedding function to embed available text.
    """
    def __init__(
        self,
        embedding: Callable[[str], list[float]],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self._store: dict[str, Document] = {}
        self._embedder = embedding
        self._chunk_size = chunk_size
        self._chunk_overlap = overlap

    def get_document(self, url: str) -> Document | None:
        """
        Retrieve a document by its URL.
        """
        return self._store.get(url)
    

    async def search(self, text: str, top_k: int = 10) -> list[Content]:
        """
        Retrieve a value from the knowledge base by specified query text.
        """
        query_embedding = await self._embed(text)
        scored = []
        for doc in self._store.values():
            for chunk in doc.content:
                if chunk.embedding is not None:
                    dist = cosine_distance(query_embedding, chunk.embedding)
                    scored.append((dist, chunk))
        scored.sort(key=lambda x: x[0])
        return [chunk for _, chunk in scored[:top_k]]

    async def add_file(self, path: str, destination: str = '/') -> None:
        """
        Read a file from disk, chunk and embed its contents, then store it under the destination URL.
        """
        file_path = Path(path)
        with file_path.open('r', encoding='utf-8') as f:
            text = f.read()

        filename = file_path.name
        dest = _normalize_destination(destination)
        document_url = f"{dest}{filename}"

        chunks = _chunk_text(
            text,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        contents = []
        for i, chunk in enumerate(chunks):
            embedding = await self._embed(chunk)
            contents.append(Content(
                content=chunk,
                embedding=embedding,
                document_url=document_url,
                chunk_index=i,
            ))

        self._store[document_url] = Document(
            document_url=document_url,
            content=contents,
        )

    async def add_directory(self, path: str, destination: str = '/', filter: Callable[[str], bool] = None) -> None:
        """
        Recursively add all files in a directory to the knowledge base under the destination URL.
        """
        dest_base = _normalize_destination(destination)
        base_path = Path(path)
        for file_path in base_path.rglob('*'):
            if not file_path.is_file():
                continue
            if filter is not None and not filter(str(file_path)):
                continue
            rel_dir = file_path.parent.relative_to(base_path)
            if rel_dir == Path('.'):
                dest = dest_base
            else:
                dest = dest_base + str(rel_dir).replace('\\', '/') + '/'
            await self.add_file(str(file_path), dest)

    def list_documents(self, path: str = '/') -> list[dict]:
        """
        List documents and sub-folders directly under specified URL path.
        """
        prefix = _normalize_destination(path)
        results = {}
        for url in self._store:
            if not url.startswith(prefix):
                continue
            remainder = url[len(prefix):]
            if not remainder:
                continue
            parts = remainder.split('/')
            name = parts[0]
            if name in results:
                continue
            if len(parts) == 1:
                results[name] = {'name': name, 'type': 'file', 'url': prefix + name}
            else:
                results[name] = {'name': name, 'type': 'directory', 'url': prefix + name}
        return list(results.values())

    async def read(self, path: str, start: int = 0, end: int = None) -> str:
        """
        Read content of the specified document path. 
        Supports optional chunking via start/end parameters.
        """
        doc = self._store.get(path)
        if doc is None:
            raise KeyError(f"Document not found: {path}")
        chunks = doc.content[start:end]
        return '\n\n'.join(chunk.content for chunk in chunks)

    async def _embed(self, text: str) -> numpy.ndarray:
        """
        Run embedding on the specified text and return it as a numpy array.
        """
        result = await self._embedder(text)
        return numpy.array(result, dtype=numpy.float32)

