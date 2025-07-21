# Torch.compile

## Contents



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
