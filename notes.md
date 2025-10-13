# Notes

### Contents
 - [Download Model without Weights](#14-Download_Model_without_Weights)
 - [Terminal](#terminal)
 - [Git](#15-Git)
 - [Server Client check](#16-Server-Client-check)

------------------------

1. Install as wheel & carry it around 
    ```
    pip uninstall -y triton
    git clone https://github.com/OpenAI/triton
    cd triton/python
    python3 setup.py bdist_wheel --dist-dir=dist
    pip install dist/*.whl // the *.whl can be saved for re-use
    ```
2. vs-code minimal debug config
   ```
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "PyDebug: vLLM bench latency",
               "type": "debugpy",
               "request": "launch",
               "python": "/usr/bin/python",
               "module": "vllm.entrypoints.cli.main",
               "console": "integratedTerminal",
               "env": {
                   "VLLM_USE_V1":"1", 
                   "VLLM_ROCM_USE_AITER":"1",
                   "VLLM_ROCM_USE_TRITON_ROPE":"1",
                   "VLLM_ENABLE_V1_MULTIPROCESSING":"0",
                   "VLLM_DISABLE_COMPILE_CACHE":"1",
                   "VLLM_LOGGING_LEVEL":"DEBUG",
               },
               "justMyCode": true,
               "args": [
                   "bench",
                   "latency",
                   "--model", 
                   "/data/models/DeepSeek-R1-dontUse/",
                   "--block-size", "1",
                   "--load-format", "dummy",
                   "--input-len", "2",
                   "--output-len", "6",
                   "--batch-size", "10",
   
                   // "--enforce-eager",
                   // "--compilation-config",
                   // "{\"cudagraph_mode\": \"FULL\", \"level\": 0, \"cudagraph_capture_sizes\": [256,248,240,40,32,24,16,8,4,2,1]}"
               ]
           }
       ]
   }
   ```

   ```
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "PyDebug: file with args",
                "type": "debugpy",
                "request": "launch",
                "python": "/usr/bin/python",
                "program": "${file}", // or "<python_script_path>.py",
                "console": "integratedTerminal",
                "env": {"AWESOME_ENV":"0"},
                "justMyCode": false,
                "args": [
                    "--test", "dequantize"
                ]
            }
        ]
    } 
    ```

    ```
    {
    "version": "0.2.0",
    "configurations": [
        {
            "name": "PyDebug: torchrun",
            "type": "debugpy",
            "request": "launch",
            "python": "/opt/conda/envs/py_3.9/bin/python",
            "program": "/opt/conda/envs/py_3.9/bin/torchrun",
            "console": "integratedTerminal",
            "env": {"AWESOME_ENV":"0"},
            "justMyCode": false,
            "args": [
                "/path/to/my_script.py",
                        "--arg1", "val1"
                    ]
                }
            ]
        } 
    ```
4. ### Terminal  
    1. Press `Enter`, `~`, `.` one after the other to disconnect from a frozen session.
    2. Use `reset` command if terminal shows weird characters on mouse clicks. They results from mouse tracking left on & session disconnects
    3. `wsl.exe --shutdown` to resolve & restart the wsl service
       ```
       Fatal Error
       Error code: Wsl/Service/E_UNEXPECTED
       press any key to continue
       ```
     4. `cd -` to navigate to previous directory
     5. Kill all bg processes by using '%': `kill -9 %`
     6. `locate` cmd to search files
         ```
         sudo apt update && sudo apt install mlocate
         updatedb -U /path/to/directory
         locate <search_string>
         ```
     7. Copying from `vi` editor to system clipboard
        ```
        1. Hold Shift and drag to select text (if tmux mouse mode is enabled via set -g mouse on)
        2. Press Ctrl + Shift + C to copy
        3. Paste anywhere with Ctrl + v
        ```

5. `"debug.inlineValues": "on"` in the settings.json. To show inline values of variables next to it while debugging.
6. Docker:
   1.   ```
        docker cp <source_path> <container_id>:<destination_path>
        docker cp container_id:/path/in/container /path/on/host
   2. asdf
7. --
8. Dump ir
   ```
    export MLIR_ENABLE_DUMP=1
    export AMDGCN_ENABLE_DUMP=1
    Always clear the Triton cache before each run by rm -rf /root/.triton/cache
    Remember to pipe the output to a file
    <cmd> > ir.txt 2>&1
    ```
9. To prevent GPU cores from creating, run your dockers with `--ulimit core=0:0`
10. To avoid wide docker ps output:
   ```
   docker ps --format="table {{.ID}}\t{{.Image}}\t{{.RunningFor}}\t{{.Status}}\t{{.Names}}"

   {{.Ports}} -> if you want it
   ```
11. Nsight System profiling <br>
    [Reference Link](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)
    ```
    nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o my_profile python main.py
    ```
    ```
    # To use mpi
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    ```
    ```
    python3 run.py --engine_dir /Projects/trt_engines/mixtral22B/tp4/ --tokenizer_dir /data/Mixtral-8x22B-Instruct-v0.1/ --max_output_len 8 --input_text "what's the capital of france?"

    python benchmark.py -m dec --engine_dir /Projects/trt_engines/trtengine_llama2_7b/ --batch_size 32 --input_output_len "32, 5" --csv
    # mixtral tp4
    mpirun -n 4 python benchmark.py -m dec --engine_dir /Projects/trt_engines/mixtral22B/tp4/ --batch_size 1 --input_output_len "512,8" --csv
    --duration -1
    --warm_up 1
    --num_runs 1
    
    ```
    A better way is to use cpp benchmarking instead of python [(link)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/README.md)
12. TRT-LLM
    - Convert mixtral's checkpoint (from examples/mixtral)
    ```
    TP=8
    python ../llama/convert_checkpoint.py --model_dir /data/Mixtral-8x22B-Instruct-v0.1/ \
            --output_dir /data/tllm_ckpt_mixtral8x22b_tp8 --dtype float16 \
            --tp_size 8 --moe_tp_size 8

    trtllm-build --checkpoint_dir /data/tllm_ckpt_mixtral8x22b_tp8 \
                 --output_dir /data/trt_engine_mixtral8x22b_tp8 \
                 --gemm_plugin float16
    
    ```
    - 

13. Python tricks
    ```python
    print("hello" "world")  # helloworld
    print("hello", "world") # hello world
    print("hello"
          "world")          # helloworld
    ```
    - Use `stack` when creating a new dimension. Use `concatenate` when merging tensors into same dims

#### 14. Download_Model_without_Weights     
  <details>
  <summary> Old way (won't work if the file store git lfs pointer) </summary>
  <br>
  
    ```
    #/bin/bash
    
    DOWNLOAD_PATH="/data/models/DeepSeek-V3"
    MODEL_REPO="https://huggingface.co/deepseek-ai/DeepSeek-V3"
    
    FILES_TO_DOWNLOAD=(
        "/inference/"
        "config.json"
        "configuration_deepseek.py"
        "model.safetensors.index.json"
        "modeling_deepseek.py"
        "tokenizer.json"
        "tokenizer_config.json"
    )
    
    mkdir -p $DOWNLOAD_PATH
    cd $DOWNLOAD_PATH
    
    git init
    git remote add origin $MODEL_REPO
    git config core.sparseCheckout true
    
    for file in "${FILES_TO_DOWNLOAD[@]}"; do
        echo $file >> .git/info/sparse-checkout
    done
    git fetch origin main
    git checkout main
    ```
  </details>

    ```
    sudo apt-get install git-lfs
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024
    git lfs pull --include "tokenizer.json"   ## regex works too.
    ```
    
#### 15. Git
- To check what vllm version & branch does the docker container have:
  ```
  pip show vllm
  git branch --contains <git_hash>  ## Shows all the branches that contain this commit
  git show <git_hash>
  ```
- One liners:
  ``` 
  git branch --contains $(pip show vllm | awk -F': ' '/^Version:/ {print $2}' | grep -oP '(?<=\+g)[0-9a-f]+')
  git show $(pip show vllm | awk -F': ' '/^Version:/ {print $2}' | grep -oP '(?<=\+g)[0-9a-f]+') | head
  ```  

#### 16. Server-Client-check
```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/data/models/Llama-3.1-8B-Instruct",
  "prompt": "The capital of France is ",
  "max_tokens": 10
}'
```
