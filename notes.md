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
                "name": "Python Debugger: Current File with Arguments",
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
3. Use `reset` command if terminal shows weird characters on mouse clicks. They results from mouse tracking left on & session disconnects 
4. asdf as
