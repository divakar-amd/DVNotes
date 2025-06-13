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
