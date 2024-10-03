# Mixture of Experts

1. [Vanilla implementation](https://github.com/pcmoritz/vllm-public/blob/fd4ea8ef5c17a8b991107402a414f6ed355d854d/vllm/model_executor/models/mixtral.py#L133)
2. `max_model_len`: max_possition_embeddings
    - Mixtral-8x7B: [32768](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/ffe1a706bacbd5abddc5ff99432ee38f7e0662fb/config.json#L12)
    - Mixtral-8x22B: [65536](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/988690dfcb44977ec67e8b7f7fb663087b4808c5/config.json#L12)
3. ### Graph Capture
    1. `max_seq_len_to_capture` 
    2.  [_verify_cuda_graph](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/config.py#L335):    
           ```python
           self.max_seq_len_to_capture = min(self.max_seq_len_to_capture, self.max_model_len)
           ```