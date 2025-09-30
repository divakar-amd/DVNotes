# Torch.compile

## Contents
- [Overview](#Overview)
- [Dynamo](#Dynamo)
- [Backend](#Backend)
- [Custom-Ops / Fake Impl](#custom-ops--fake-impl)
- [Torch.compile in vLLM-V1](#Torchcompile-in-vLLM-V1)


#### Overview
- Torch.compile has a frontend and a backend
  - Fortend: Dynamo. Traces the torch modules and creates an FX graph.
  - Backed: multiple backend that take the FX graph from Dynamo and spits out an optimised code. Defaut it "Inductor"
 
#### Dynamo
- Operates on python bytecode (which is 1-level lower than the human readable python code)
- It looks for PyTorch code in the Python code.
- Dynamo support **dynamic** input shapes.

#### Backend
- Once the computation graph is obtained from the previous steps, it's all about graph optimization.
- There is also an "eager" mode backend if you want to skip the graph optimization step. Handy for debugging.

#### Custom-Ops / Fake Impl
- Pytorch allows using custom ops via `torch.library.custom_ops()`
- Now, if are registering a custom-op AND the op returns something AND you are using torch.compile --> You need to write a "Fake" impl of the function. The "fake" impl provides all the necessary info about the input & output tensors for it to work smoothly with Dynamo / FX tracing. Think about shape/dtype/device. If the op returns nothing i.e. **None**, then there's no need for the "fake" impl.
- If you are not using torch.compile, you can simply register a custom-op without worrying about the "fake" impl (because no FX tracing / Dynamo is involved)
- [Resource link](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- Important Update: By default, for PIECEWISE compilation, custom ops are disabled and vllm falls back to Native implementation. To use custom ops, they need to be specified in the compilation config.
  ```
  VLLM_ROCM_USE_TRITON_ROPE=1  .... --compilation-config='{"custom_ops":["+rotary_embedding"]}'
  ``` 

#### Torch.compile in vLLM-V1
- Invoked using `@support_torch_compile` decorator on model's class
- The first run that vLLM does is estimating the available memory for KV-cache and calling the model's `forward()` method. This when the torch.compile gets kicked-in and that's because model's forward method call the decorator's `__call__` method. Read further for the input shapes...
- So, torch.compile's Dynamo will do the "tracing" and build an FX graph **before** estimating the kv-cache size. Now, the shape we use for kv-cache estimation is the "biggest" possible input/batch sizes for the given model. e.g. We use `max_num_batched_tokens` which defaults to `16384` for llama-70B.
  - Torch.compile support dynamic shapes but...
  - When tracing, torch.compile will pick the very first batch-size to capture the FX graph. The shapes are marked as "symbolic" and hence dynamic.
  - But, if there's a branchking based on this batch-size, it will actually fetch the value of this symbolic shape and follow a particular branch.
  - Hence, avoid branching based on batch-size or number of elements. -Or- wrap the condition under a torch custom ops
  - There is not torch.compile involved when using **eager-mode** in vLLM. [Link](https://github.com/vllm-project/vllm/blob/80141bbf2f1b8b0beaac097f94923f95773734ef/vllm/config/__init__.py#L3531-L3536)
