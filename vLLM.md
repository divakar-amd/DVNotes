# vLLM Notes


# Contents
 - [Entrypoint](#Entrypoint) 
 - [Formatting](#Formatting)
 - [Extra-Decode](#Extra-Decode)


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
