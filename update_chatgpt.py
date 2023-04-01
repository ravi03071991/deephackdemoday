                        Updated gpt-index/ llama-index code base:

```
from gpt_index import (
    GPTTreeIndex,
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    Response
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from gpt_index.evaluation import ResponseEvaluator
import pandas as pd
pd.set_option('display.max_colwidth', 0)

# gpt-3.5-turbo
llm_predictor_gpt35 = LLMPredictor(llm=ChatOpenAI(
    temperature=0, model_name="gpt-3.5-turbo"))
service_context_gpt35 = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_gpt35)

documents = SimpleDirectoryReader('../test_wiki/data').load_data()

tree_index = GPTTreeIndex.load_from_disk('../test_wiki/index.json')

# create vector index
vector_index = GPTSimpleVectorIndex.from_documents(
    documents,
    service_context=ServiceContext.from_defaults(chunk_size_limit=512)
)
vector_index.save_to_disk('../test_wiki/simple_vector_index.json')

vector_index = GPTSimpleVectorIndex.load_from_disk(
    '../test_wiki/simple_vector_index.json')
```

In this updated version of the code, we replaced gpt-3 (davinci) with gpt-3.5-turbo, which is a more advanced model from OpenAI. The core functionality of loading an index, creating a vector index, and saving/loading it to/from disk remains unchanged, but the improvements in gpt-3.5-turbo will help in generating better chat responses.