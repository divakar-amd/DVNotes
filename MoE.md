# Mixture of Experts

1. [Vanilla implementation](https://github.com/pcmoritz/vllm-public/blob/fd4ea8ef5c17a8b991107402a414f6ed355d854d/vllm/model_executor/models/mixtral.py#L133)
2. `max_model_len`: [max_possition_embeddings](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/config.py#L1686)
    - Mixtral-8x7B: [32768](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/ffe1a706bacbd5abddc5ff99432ee38f7e0662fb/config.json#L12)
    - Mixtral-8x22B: [65536](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/988690dfcb44977ec67e8b7f7fb663087b4808c5/config.json#L12)
3. ### Graph Capture Size
    1. `max_seq_len_to_capture` : [name terminology](https://github.com/vllm-project/vllm/pull/4518)
    2.  [_verify_cuda_graph](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/config.py#L335):    
           ```python
           self.max_seq_len_to_capture = min(self.max_seq_len_to_capture, self.max_model_len)
           ```
    3. [_get_graph_batch_size](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/worker/model_runner.py#L1876) : Returns the padded batch size given actual batch size. Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT, 2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
4. ### Graph usage condition
    1. `_use_captured_graph` : [link](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/worker/model_runner.py#L713-L722)
    2. <details>
            <summary>print log for Mixtral-8x22B</summary>
            <br>
            <pre> --input-len 8192 --output-len 3 --batch-size 32  </pre>
            <pre>max_seq_len_to_capture is set to 8192+256=8448</pre>
            <pre>
                decode_only: False && not enforce_eager: False
                ,  batch_size: 65536 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: 0, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 65536 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = False
                decode_only: False && not enforce_eager: False
                ,  batch_size: 65536 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: 0, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 65536 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = False
                decode_only: False && not enforce_eager: False
                ,  batch_size: 65536 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: 0, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 65536 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = False
                decode_only: False && not enforce_eager: False
                ,  batch_size: 65536 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: 0, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 65536 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = False
                decode_only: True && not enforce_eager: False
                ,  batch_size: 32 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: **8193**, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 32 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = True
                decode_only: True && not enforce_eager: False
                ,  batch_size: 32 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: **8194**, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 32 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = True
                decode_only: True && not enforce_eager: False
                ,  batch_size: 32 <= _BATCH_SIZES_TO_CAPTURE: 8192
                ,  max_decode_seq_len: **8195**, max_encoder_seq_len: 0 <=  max_seq_len_to_capture: 8448
                   batch_size: 32 <= max_batchsize_to_capture: 256
                   --> result (_use_captured_graph) = True
            </pre>
        </details>

5. Chunked prefill gets enabled automatically for model_len > 32k: [link](https://github.com/vllm-project/vllm/blob/83caf35e082b2657dce5f71ff965a13653a763b0/vllm/engine/arg_utils.py#L929-L931)
6. Decode latency IS affected by prefill length.
   - BS=240, output=200
   
    | Input-len | Decode latency |
    |-----------|----------------|
    | 512       | 63ms           |
    | 1024      | 66ms           |
    | 2048      | 69ms           |
   <details>
            <summary>Reason </summary>
            <br>
            <pre> kv cache size is bigger for larger context length. Hence, paged_attn kernel takes more time!  </pre>
            </pre>
        </details>
