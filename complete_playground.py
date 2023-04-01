# Hide INFO logs regarding token usage, etc
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.indices.vector_store import GPTSimpleVectorIndex
from llama_index import download_loader
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin'])

indices = [GPTSimpleVectorIndex.from_documents(
    documents), GPTTreeIndex.from_documents(documents)]
