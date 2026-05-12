## Prefill Node
HIP_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve /data/models/Llama-3.1-8B/ --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' --port 8100

## Decode Node
HIP_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve /data/models/Llama-3.1-8B/ --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' --port 8201

## Proxy server
python disagg_prefill_proxy_server.py --port 8000 --prefill-url http://localhost:8100 --decode-url http://localhost:8201

## Client cmd
curl -X POST http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
      "model": "/data/models/Llama-3.1-8B/",
      "prompt": "write an essay on Paris ",
      "max_tokens": 100,
      "temperature": 0
    }'
