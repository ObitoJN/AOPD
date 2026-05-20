export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

which python

model_name_or_path=/path/to/your/teacher/model
served_model_name=Qwen3-32B


NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))


tensor_parallel_size=$GPUS_PER_NODE

echo $model_name_or_path
echo $served_model_name


echo "服务已部署，访问地址为：http://$(hostname -I | awk '{print $1}'):8000"
python3 -m vllm.entrypoints.openai.api_server --model ${model_name_or_path} \
    --api-key your_api_key \
    --tensor-parallel-size ${tensor_parallel_size} \
    --served-model-name ${served_model_name} \
    --gpu-memory-utilization 0.6 \
    --max-logprobs 48 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --swap-space 50

sleep 10
