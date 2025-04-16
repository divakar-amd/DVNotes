# Installation steps to reproduce


# Contents
 - [TensorRT-LLM](#TensorRT-LLM) 
 - [Nsight systems](#Nsight%20systems%20(nsys))
 - [vLLM-cuda](#vLLM-cuda)


## TensorRT-LLM
Steps for setting up trt-llm from source. [ref link](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html)
```console
# build the docker
make -C docker build CUDA_ARCHS="90-real"

# run the docker
docker run --name dv_trt_src --gpus all -i -d --ipc=host --entrypoint /bin/bash  -v /data/:/data -v /home/divverma/Projects/:/Projects --volume /home/divverma/Projects/TRT_LLM_2/TensorRT-LLM:/code/tensorrt_llm  --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" --env "CCACHE_BASEDIR=/code/tensorrt_llm"   --workdir /code/tensorrt_llm  --tmpfs /tmp:exec  tensorrt_llm/devel:latest

# build the trtllm wheel inside the docker
python3 ./scripts/build_wheel.py --cuda_architectures "90-real" --trt_root /usr/local/tensorrt --benchmarks --clean

# Deploy TensorRT-LLM in your environment.
pip install ./build/tensorrt_llm*.whl

```

## Nsight systems (nsys)
```console
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install -y nsight-systems-cli
pip install nvtx
```

## vLLM-cuda
If you only need to change Python code, you can build and install vLLM without compilation. ([link]([url](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html?device=cuda#build-wheel-from-source)))
```
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
