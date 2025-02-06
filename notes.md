# Notes

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
                "name": "PyDebug: file with args",
                "type": "debugpy",
                "request": "launch",
                "python": "/opt/conda/envs/py_3.9/bin/python",
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
3. Use `reset` command if terminal shows weird characters on mouse clicks. They results from mouse tracking left on & session disconnects 
    3.1 Press `Enter`, `~`, `.` one after the other to disconnect from a frozen session.
4. `"debug.inlineValues": "on"` in the settings.json. To show inline values of variables next to it while debugging.
5. Docker:
   1.   ```
        docker cp <source_path> <container_id>:<destination_path>
        docker cp container_id:/path/in/container /path/on/host
   2. asdf
6. Kill all bg processes by using '%': `kill -9 %`
7. Dump ir
   ```
    export MLIR_ENABLE_DUMP=1
    export AMDGCN_ENABLE_DUMP=1
    Always clear the Triton cache before each run by rm -rf /root/.triton/cache
    Remember to pipe the output to a file
    <cmd> > ir.txt 2>&1
    ```
8. To prevent GPU cores from creating, run your dockers with `--ulimit core=0:0`
9. To avoid wide docker ps output:
   ```
   docker ps --format="table {{.ID}}\t{{.Image}}\t{{.RunningFor}}\t{{.Status}}\t{{.Names}}"

   {{.Ports}} -> if you want it
   ```
10. Nsight System profiling <br>
    [Reference Link](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)
    ```
    nsys profile --stats=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o my_profile python main.py
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
11. Python tricks
    ```python
    print("hello" "world")  # helloworld
    print("hello", "world") # hello world
    print("hello"
          "world")          # helloworld
    ```
12. vLLM pre-commit formatting
    ```
    pip install -r requirements-lint.txt && pre-commit install
    ```


