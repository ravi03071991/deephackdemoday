```python
from gpt_index import (
    GPTTreeIndex,
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    Response
)
from langchain.llms import OpenAI
from gpt_index.evaluation import ResponseEvaluator

# gpt-3 (davinci)
llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(
    temperature=0, model_name="text-davinci-003"))
service_context_gpt3 = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_gpt3)

documents = SimpleDirectoryReader('./data').load_data()

tree_index = GPTTreeIndex.load_from_disk('./tree_index.json')

# create vector index
vector_index = GPTSimpleVectorIndex.from_documents(
    documents,
    service_context=ServiceContext.from_defaults(chunk_size_limit=512)
)
vector_index.save_to_disk('./simple_vector_index.json')

vector_index = GPTSimpleVectorIndex.load_from_disk(
    './simple_vector_index.json')

# Evaluation
def evaluate_response(query: str, index, mode: str) -> str:
    response = index.query(query)
    evaluator = ResponseEvaluator(mode=mode, service_context=service_context_gpt3)
    evaluation = evaluator.evaluate(response)
    return evaluation

query_str = "your query here"
mode = "context_response"  # Choose the evaluation mode ("context_response" is the only available mode)

evaluation_result = evaluate_response(query_str, vector_index, mode)
print(f"Evaluation: {evaluation_result}")
```

In the code above, we import the necessary modules and initialize the GPT-3 predictor, load the data and gpt/index files, and create a helper function `evaluate_response` to perform the evaluation on a given query. The function takes a query string, an index (e.g., vector_index), and an evaluation mode as input and returns the evaluation result.