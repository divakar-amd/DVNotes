# vLLM Notes


# Contents
 - [Entrypoint](#Entrypoint) 
 - [Formatting](#Formatting)
 - [Extra-Decode](#Extra-Decode)
 - [Profile-using-nsys](#Profile-using-nsys)
 - [Namings](#Namings)


### Entrypoint
- vLLM upstream containers have set openai servers as the default entrypoint [(link)](https://github.com/vllm-project/vllm/blob/bc1bdecebf76cca0dfafe4924d529b30c8a24795/Dockerfile#L278). Use the following when launching docker container
  ```
  docker run --entrypoint /bin/bash ....
  ```

### Formatting
- Use pre-commit for formatting. Run this before commits.
    ```
    pip install -r requirements-lint.txt && pre-commit install
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
```
if profile_dir:
            torch.cuda.cudart().cudaProfilerStart()
            nvtx.push_range("dv_llm_generate")
            llm_generate()
            nvtx.pop_range()
            torch.cuda.cudart().cudaProfilerStop()
```
```
nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o nsys_vllm_llama7B_in128_bs64_out1 \
 python3 benchmark_latency_nsys.py --model /data/llama-2-7b-chat-hf/ --load-format dummy --dtype float16 --input-len 128 --output-len 1 --batch-size 64 \
 --num-iters 1 --num-iters-warmup 10 --disable-async-output-proc --enforce-eager --profile
```
```
# generate csv stats table for the gpu kernels
nsys stats trace_file.sqlite  --report cuda_gpu_kern_sum --format csv --output out.csv
```

### Namings

1. `max_seq_len_to_capture`: Maximum sequence len covered by CUDA graphs. If greater than this, fall-back to eager-mode.
   - `max_seq_len_to_capture = min(max_seq_len_to_capture, max_model_len)` ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L635))
3. `max_model_len`: Maximum length of a sequence (including **prompt and output**). If None, will be derived from the model. ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L2410)).
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
    
5. `max_num_batched_tokens`
6. `max_num_seqs`
7. `max_seq_len`
8. `max_num_batched_tokens`: This comes handy when using chunked-prefill. Only when chunked-prefill is enabled, this value can be smaller than `max_model_len`. If chunked-prefill is disabled, this value is `max(max_model_len, 2048)` ([link](https://github.com/vllm-project/vllm/blob/d84cef76eb9e16190cfdd97ae24511c8c819f179/vllm/config.py#L1546))
