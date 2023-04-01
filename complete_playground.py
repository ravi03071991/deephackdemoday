```python
# Import necessary libraries and modules
import logging
from llama_index import download_loader
from llama_index.indices.vector_store import GPTSimpleVectorIndex
from llama_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.playground.base import Playground

# Hide INFO logs regarding token usage, etc
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Load data
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin'])

# Create indices
indices = [GPTSimpleVectorIndex.from_documents(documents),
           GPTTreeIndex.from_documents(documents),
           GPTListIndex.from_documents(documents)]

# Initialize the Playground with the created indices
playground = Playground(indices)

# Query to compare
query_text = "What is the population of Berlin?"

# Compare the output of the indices
result = playground.compare(query_text, to_pandas=True)

# Print the result
print(result)
```

This playground code imports the necessary libraries and modules, loads the Wikipedia data for the page "Berlin", creates three different indices (GPTSimpleVectorIndex, GPTTreeIndex, and GPTListIndex) using the provided gpt-index/llama-index code base, initializes a Playground using these indices, and then uses the `compare` function of the Playground class to compare the outputs for a given query. The result is printed as a pandas DataFrame.