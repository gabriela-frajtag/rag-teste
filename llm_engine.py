from llama_cpp import Llama
import os

llm = Llama(model_path="models/tinyllama.gguf", n_ctx=2048)

def ask_llm(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = llm(prompt, max_tokens=256, stop=["\n", "</s>"])
    return output["choices"][0]["text"].strip()
