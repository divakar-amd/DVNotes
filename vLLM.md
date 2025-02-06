# vLLM Notes


# Contents
 - [Entrypoint](#Entrypoint) 
 - [Formatting](#Formatting) 


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
