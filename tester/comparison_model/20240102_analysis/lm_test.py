from langchain.llms import Ollama

llm=Ollama(model="llama2")
# llm=Ollama(model="llava")
# llm=Ollama(model="mistral")
# llm=Ollama(model="vicuna")

res = llm.predict('ababs')
print(res)