MODEL=/data/models/Llama-3.1-70B-Instruct
REQUEST_RATES=(1 5 7 9)
TOTAL_SECONDS=120
RESULTS_DIR="bench_results"

mkdir -p "$RESULTS_DIR"

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
    NUM_PROMPTS=$(($TOTAL_SECONDS * $REQUEST_RATE))

    echo ""
    echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
    echo ""

    python3 benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --request-rate $REQUEST_RATE \
        --random-input-len 4096 \
        --random-output-len 256 \
        --num-prompts $NUM_PROMPTS \
        --ignore-eos --seed $REQUEST_RATE \
        --seed 42 \
        > "${RESULTS_DIR}/result_${REQUEST_RATE}qps.txt" 2>&1
done
