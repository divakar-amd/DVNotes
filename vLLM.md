# vLLM Notes


# Contents
 - [Entrypoint](#Entrypoint) 
 - [Formatting](#Formatting)
 - [Extra-Decode](#Extra-Decode)
 - [Profile-using-nsys](#Profile-using-nsys)
 - [Namings](#Namings)
 - [Graph-Capture](#Graph-Capture)
 - [KV-Cache](#KV-Cache)
 - [Self-Attention](#Self-Attention)
 - [Warm-ups](#Warm-ups)
 - [Debug](#Debug)



### Entrypoint
- vLLM upstream containers have set openai servers as the default entrypoint [(link)](https://github.com/vllm-project/vllm/blob/bc1bdecebf76cca0dfafe4924d529b30c8a24795/Dockerfile#L278). Use the following when launching docker container
  ```
  docker run --entrypoint /bin/bash ....
  ```

### Formatting
- Use pre-commit for formatting. Run this before commits.
    ```
    pip install -r requirements/lint.txt && pre-commit install
    ```
### Extra-Decode
- When using `--output-len=1` (in an attempt to capture prefill only), an extra Decode step happens owing to **Asynchronous Output Processing** [(link)]([url](https://blog.vllm.ai/2024/09/05/perf-update.html)). Hence, you would see an extra Graph Launch for the decode in the profiler trace.
- Use `--disable-async-output-proc` when you want to profile only the Prefill part. This would void the extra graph launch for the Decode step.
- Or, `--enforce-eager` can also be used.

### Prefill
```
python3 benchmark_latency.py --model /data/llama-2-7b-chat-hf/ -tp 1 --dtype float16 --load-format dummy \
--input_len 2049 --output_len 1 --batch_size 2 --num-iters 1 --num-iters-warmup 0 \
--profile --profile-result-dir vllm_benchmark_result_llama2-7B_in2049_out1_bs2_tp1_eagerMode_warmup0_withStack_try8_noAsyncOut \
--disable-async-output-proc --enforce-eager
```

### Profile-using-nsys
- For tp>1, use `--trace-fork-before-exec=true`
```
if profile_dir:
            torch.cuda.cudart().cudaProfilerStart()
            nvtx.push_range("dv_llm_generate")
            llm_generate()
            nvtx.pop_range()
            torch.cuda.cudart().cudaProfilerStop()
```
```
## llama-7B (tp=1)
nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o nsys_vllm_llama7B_in128_bs64_out1 \
 python3 benchmark_latency_nsys.py --model /data/llama-2-7b-chat-hf/ --load-format dummy --dtype float16 --input-len 128 --output-len 1 --batch-size 64 \
 --num-iters 1 --num-iters-warmup 10 --disable-async-output-proc --enforce-eager --profile
```
```
## Mixtral8x7b (tp=2)
nsys profile --stats=true -w true -t cuda,nvtx,cudnn,cublas --trace-fork-before-exec=true  --capture-range=cudaProfilerApi  -x true -o nsys_vllm_mixtral8x7B_tp2_in128_bs64_out1  python3 benchmark_latency_nsys.py --model /data/Mixtral-8x7B-v0.1/  --load-format dummy --dtype float16 \
--input-len 128 --output-len 1 --batch-size 64  --num-iters 1 --num-iters-warmup 10 --disable-async-output-proc --enforce-eager --profile -tp 2

```
```
# generate csv stats table for the gpu kernels
nsys stats trace_file.sqlite  --report cuda_gpu_kern_sum --format csv --output out.csv
```

### Namings

1. `max_seq_len_to_capture`: Maximum sequence len covered by CUDA graphs. If greater than this, fall-back to eager-mode.
   - `max_seq_len_to_capture = min(max_seq_len_to_capture, max_model_len)` ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L635))
3. `max_model_len`: Maximum length of a sequence (including **prompt and output**). If None, will be derived from the model. ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L2410)). E.g.: llama2-7B, config.json has `"max_position_embeddings": 4096,`
      ```
      possible_keys = [
           # OPT       "max_position_embeddings", 
           # GPT-2     "n_positions",
           # MPT       "max_seq_len",
           # ChatGLM2  "seq_length",
           # Command-R "model_max_length",
           # Whisper   "max_target_positions",
           # Others
             "max_sequence_length",
             "max_seq_length",
             "seq_len",
      ```
    
5. `max_num_batched_tokens`:  This comes handy when using chunked-prefill. Only when chunked-prefill is enabled, this value can be smaller than `max_model_len`. If chunked-prefill is disabled, this value is `max(max_model_len, 2048)` ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L1546))
6. `max_num_seqs`
7. `max_seq_len`

### Graph-Capture
1. Graph capture only happens for Decode phase (i.e. not for Prefill)
2. This is initiated after the KV-Cache memory profiling, during the engine "warmup"
3. Before capturing the graph for each Batch Size, some warm-up runs are done. This takes care of stuff like torch.compile or triton.autotune to avoid including extral kernel launches in the graph capture.
4. For each decoding run, an appropriate graph is picked depending on the batch_size and "replayed".
5. Memory consumed by all the graph-captures is actually quite small. e.g. `Deepseek-R1 takes only 0.51GB for graphs`. This is because, a captured graph pretty much stores only the pointer to the kernels and the arguments which have to be fixed (the arguments can be pointers too!)

### KV-Cache

- [Entrypoint]([url](https://github.com/vllm-project/vllm/blob/d374f04a337dbd4aab31484b6fa2d4a5f20c2116/vllm/engine/llm_engine.py#L277)) is inside LLMEngine.__init__() 
- KV-cache is built for each layer of the model. Each layer has a set of `q_proj, k_proj, v_proj and o_proj weights`. E.g. [Llama7B HF link](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/01c7f73d771dfac7d292323805ebc428287df4f9/model.safetensors.index.json#L13-L17)
- `num_attention_heads` vs `num_key_value_heads`: The segregation is for **Grouped Query Attention (GQA)**. If `num_key_value_heads` (say 16) is less than `num_attention_heads` (say 32), this means Queries are grouped into size of (= 32/16 = 2), where each group shares a single key_value head. e.g:

| Model | n_layers    | n_heads | n_KV_heads | d_head | d_model | Attention |
| ----- | ----------- | ------- | ---------- | ------ | ------- | --------- |
| Llama-2-7B  | 32      | 32         | 32     | 128     | 4096      | MHA |
|Mistral-7B	  | 32	     | 32	        | 8     	| 128    	| 4096     	| GQA |
   
1. Determine number of blocks for KV cache. Both GPU cache & swap CPU cache.
    - There are 2 modes to handle preemption of KV-cache: Recompute (default) and Swap
    - Recomputation incurs lower overhead than Swap
    - CPU KV-cache is only use in "swap" mode. ([details]([url](https://github.com/vllm-project/vllm/issues/2853#issuecomment-1943920316))). GPU<->CPU cache swapping has been deprecated in V1 (vllm).
    - Swap mode is only enabled when more than 1 sequences are running per sequence group (e.g. beam search).
  

    - 1.1 Do a profile run
       - Run with the max possible input size to get peak memory usage.
       - KV scales calculation is disabled
    - 1.2 Total memory for KV Cache = TotalGPUMem * mem_util_cofig(e.g. 90%)  -  non_kv_cache_mem (e.g. weights, NCCL etc.)
    - 1.3 Calculate cache block size and total number of cache blocks. Size of 1 cache_block: `2 * num_heads * head_size * num_layers * block_size`
    - Each cache_block has a `block_size` which is hard-coded to 16. ([e.g.](https://github.com/vllm-project/vllm/blob/ce20124671cf4580627089e02f391cc95747939f/vllm/platforms/cuda.py#L145))
  
2. Initialize cache
   - Max concurrency (to get a rough estimate of max batch size). [(PR link)](https://github.com/vllm-project/vllm/pull/8831)
   - Checks for cache size, including: max_model_length should not exceed the max no. of tokens that can be stored in the KV cache. `assert max_model_length <= num_gpu_blocks * block_size(=16)`
   - So far, only the no. of cache blocks and size have been calculated. The actual GPU memory for cache will be allocated in this step.
   - kv_cache_shape for each layer: `(2, num_gpu_kvcache_blocks, block_size * num_kv_heads * headsize)`
   - kv_cache = `[ torch.zeros(kv_cache_shape) for _ in num_layers ]`
   - For each layer, the attn kv-cache tensors are bind to the respective kv_cache memory
  
3. Warm-up model
   - Capture the cuda graphs for certain batch sizes. These cuda graphs are used during decoding.
   - Note: Cuda graph capture also takes some gpu memory and since this graph capture happens _after_ the KV-cache size calculations, this means the graph capture size is _Not_ included in the gpu_mem_utilization limits. However, it seems that it doesn't take too much space. e.g.: `llama7b (tp=1) graphs take only 0.28 GB.`
 
- ##### KV-Cache during Prefill phase
    - The prompts are passed to "add_request"
    - During prefill, the keys & values of the prefill tokens are used to 'fill' the kv-cache. However, the flash-attn kernel itself doesn't need to use the kv-cache, it already has what it needs.
- ##### KV-Cache during Decode phase
    - __
- ##### Example:
  ```
  ---For 1 Layer---
  kv_cache.shape: torch.Size([2, 20262, 65536])  # [k+v, total_kv_blocks, 16 * num_heads * head_dim ]  # 16=block_size
  key.shape:      torch.Size([10, 32, 128])      # [prefill_tokens, num_head, head_size]
  value.shape:

  Total kv_blocks needed to update this case:    key.numel() / kv_block_size = (10*32*128)/(16*32*128)
  ```

### Self-Attention
- Q, K are first massaged with rotary_embedding before the Attn. ([link](https://github.com/vllm-project/vllm/blob/fd8e055ffba508e094cd1793e49bbdc5e53b7266/vllm/model_executor/models/llama.py#L203))
- How is the `positions` vector determined here though?
- Shapes
  ```
  query:    [num_tokens, num_heads * head_size]
  key:      [num_tokens, num_kv_heads * head_size]
  value:    [num_tokens, num_kv_heads * head_size]
  kv_cache: [2, num_blocks, block_size * num_kv_heads * head_size]
  ```

### Warm-ups
- During engine initialisation in V1, there are 4 model passes:
  1. To find the max size of kv-cache. This pass passes an empty kv-cache and finds out how much of the remaining memory can be allocated to kv-cache
  2. For torch compile. KV-caches at this point have been already initiated. There will 3 "types" of torch compile graph and a total of n=num_layers torch compile graphs. The first and last layer graphs are different for the middle graphs.
  3. Cuda graph capture (with a warmup first). Cuda graph captures are done for all the specified sizes. [link](https://github.com/vllm-project/vllm/blob/a1cc9f33a32eef4550daccdc76aefc1baf7bc35d/vllm/v1/worker/gpu_worker.py#L240-L244)
  4. A yet another run to warm-up the sampler. [link](https://github.com/vllm-project/vllm/blob/a1cc9f33a32eef4550daccdc76aefc1baf7bc35d/vllm/v1/worker/gpu_worker.py#L246-L250)
 

### Debug
- `VLLM_LOGGING_LEVEL=DEBUG`
- For V1 debugging: `VLLM_ENABLE_V1_MULTIPROCESSING=0` +  `breakpoint()` inside the python file. Launch as a normal python cmd.
- ```
  VLLM_ENABLE_V1_MULTIPROCESSING=0  VLLM_LOGGING_LEVEL=DEBUG VLLM_USE_V1=1 python benchmark_latency.py --model /data/Llama-2-7b-hf/ --load-format dummy -tp 1 --input-len 10 --output-len 10 --batch-size 10 --compilation-config '{"use_inductor": "False", "custom_ops": ["all"], "cudagraph_capture_sizes": [128]}'
  ```
