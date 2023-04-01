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

# gpt-3 (davinci)
llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(
    temperature=0, model_name="text-davinci-003"))
service_context_gpt3 = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_gpt3)

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
