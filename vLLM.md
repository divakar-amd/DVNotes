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
