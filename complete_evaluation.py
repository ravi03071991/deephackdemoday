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

# Sample query to test the evaluation
query = "How many continents are there in the world?"

tree_response = tree_index.query(query)
vector_response = vector_index.query(query)

# Instantiate the ResponseEvaluator
evaluator = ResponseEvaluator(service_context=service_context_gpt3)

# Evaluate the responses
evaluation_tree_response = evaluator.evaluate(tree_response)
evaluation_vector_response = evaluator.evaluate(vector_response)

print(f"Tree Response Evaluation: {evaluation_tree_response}")
print(f"Vector Response Evaluation: {evaluation_vector_response}")
```

This is the complete evaluation code using the provided gpt-index/llama-index code base and documentation. It includes loading the GPTTreeIndex and GPTSimpleVectorIndex, running a query on each, and evaluating the responses using the provided ResponseEvaluator from the gpt_index.evaluation module. Make sure to replace the `query` variable with your desired query.