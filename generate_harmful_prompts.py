from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",  
    n_ctx=4096,
    n_threads=8
)

output = llm("Explain gradient descent in one sentence.")
print(output["choices"][0]["text"])
