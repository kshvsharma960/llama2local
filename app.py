from llama_cpp import Llama

# define n_ctx manually to permit larger contexts
LLM = Llama(model_path="llama-2-7b-chat.ggmlv3.q8_0.bin", n_gpu_layers=28, n_threads=6, n_ctx=3584, n_batch=521, verbose=True)

# create a text prompt
prompt = "Q: Why are Jupyter notebooks difficult to maintain? A:"

# set max_tokens to 0 to remove the response size limit
output = LLM(prompt, max_tokens=0)

# display the response
print(output["choices"][0]["text"])