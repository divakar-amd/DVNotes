# About LLMs and related things

## Table of Contents
**[1. Quantization](#quantization)**<br>
**[2. Training](#training)**<br>
**[3. Tokenization](#tokenization)**<br>
**[4. Temperature, top_p, top_k](#temperature)**<br>
**[5. Tensor Parallelism](#TensorParallelism)**<br>


<br> 

### Quantization
- Consider a 2D matrix MxN where M = num_tokens and N = num_channels
- 3 modes:
    - **Per-tensor** scaling: Use a single scaling factor for the entire 2D matrix
    - **Per-token** scaling: Use "M" scaling factors, one for each row
    - **Per-channel** scaling: Use "N" scaling factors, one for each channel


| Name        | precision    |                                                  |
|-------------|--------------|--------------------------------------------------|
| SmoothQuant | Int8 (W8A8)  | Requires pre-processing the weights              |
| GPTQ, AWQ   | W4A16        | AWQ: Preserves & doesn't quantize all the weights |
| Weight only | W4A16, W8A16 |                                                  |

### Training
- 3 stages:
    - **Pre-training**: Most compute heavy. Model learns broad/basic things like language structure.
    - **Post-training**: Fine-tuning, domain specific training.
    - **Instruct-training**: Train the model to better parse a human's prompt and understand what's asked. e.g.
      ```
      Prompt: "What's the capital of France?"

      # without instruct-training
      "What's the capital of Canada? What's the capital of Spain? ... "
      
      # with instruct-tuning
      "The capital of France is Paris....."
      ```


### Tokenization
- All text characters have associated Unicode values. Size of total Unicode values is about 150k. Not possible to train with this big vocabulary size though.
- Some common ways to encode the unicode text include UTF-8, UTF-16 etc. Since the unicode values are reduced to byte level, the range of an ecoded character is [0, 256) which solve the OOVocabulary issue. However, this techniques usually results in long encoded sequeces, making it unfeasible to train the model (capturing prev context withing these long seq is also difficult).
- BPE (byte pair encoding) is a middle ground for the above issues. Frequently occuring bytes are merged together and added to vocab. Training a BPE encoder means merging these frequent occurences together and adding to vocabulary and hence, reducing sequence length.


### Temperature, top_p, top_k
- Temperature: controls randomness. Higher temp means more randomness when selecting output probabilities
- `top_p`: controls output vocab size. Higher top_p means output token is selected from larger vocab pool. Lower value means less but more probable ones.
- `top_k`: controls output vocab size. Higher top_k means output token is chosen from a larger vocab pool. Lower value means less vocab but highly probable ones.
- In both `top_p` & `top_k`, the vocab tokens are sorted based on their probabilities.


#### TensorParallelism
- Both Column Parallel and Row Parallel are used to split the weights for GEMMs. [vLLM blog post](https://blog.vllm.ai/2025/02/17/distributed-inference.html)
- Up Projection -> Column Parallel. E.g. [llama gate up](https://github.com/ROCm/vllm/blob/f94ec9beeca1071cc34f9d1e206d8c7f3ac76129/vllm/model_executor/models/llama.py#L76)
- Down Projection -> Row Parallel. E.g. [llama gate down](https://github.com/ROCm/vllm/blob/f94ec9beeca1071cc34f9d1e206d8c7f3ac76129/vllm/model_executor/models/llama.py#L83)
- This means ["model.layers.0.mlp.**up_proj**.weight"](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/pytorch_model.bin.index.json#L11) will be split column-wise when using tp>1 whereas, ["model.layers.0.mlp.**down_proj**.weight"](url) will be split row-wise when using tp>1.
