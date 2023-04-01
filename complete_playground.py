```python
# Hide INFO logs regarding token usage, etc
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from llama_index import download_loader
from llama_index.indices.vector_store import GPTSimpleVectorIndex
from llama_index.indices.tree.base import GPTTreeIndex

from gpt_index.playground.base import Playground
from gpt_index.readers.schema.base import Document

# Load the WikipediaReader
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin'])

# Create indices using the loaded documents
indices = [
    GPTSimpleVectorIndex.from_documents(documents),
    GPTTreeIndex.from_documents(documents)
]

# Initialize Playground with the created indices
playground = Playground(indices)

# Define and compare a sample query
query_text = "What is the capital of Germany?"
comparison_result = playground.compare(query_text)

# Print the comparison results
print("\nComparison Results:")
print(comparison_result)

# Run the Playground interactively (optional)
# playground.interact()
```

In this code snippet, we first import the necessary modules and classes, such as `logging`, `download_loader`, `GPTSimpleVectorIndex`, `GPTTreeIndex`, and `Playground`. Then, we load the `WikipediaReader`, create the two index types using the loaded documents, and initialize the `Playground` with these indices.

After that, we define a sample query and compare the outputs of the different indices. The comparison results are then printed. Optionally, you can uncomment the last line to run the Playground interactively.