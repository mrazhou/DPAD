set -x
echo $(which python)

export VLLM_ATTENTION_BACKEND=XFORMERS
# export LD_LIBRARY_PATH=xxx/envs/seg_zero/lib/

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
RUN_NAME=$1/$(basename "$0" .sh)

gpus=8

start_time=$(date +%s)
mkdir -p ./workdir/${RUN_NAME}

python -m verl.trainer.main \
    config=training_scripts/seg_zero_3b.yaml \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=5.0e-3 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=./workdir/${RUN_NAME} | tee ./workdir/${RUN_NAME}/output.log

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "================================================"
echo "Evaluating the model..."
last_step=$(cat ./workdir/${RUN_NAME}/latest_checkpointed_iteration.txt)
bash evaluation_scripts/eval_all.sh ./workdir/${RUN_NAME}/global_step_${last_step}/actor



echo "================================================"
echo "Training completed in $((duration / 86400)) days and $((duration % 86400 / 3600)) hours and $((duration % 3600 / 60)) minutes"
echo "Saved checkpoint to ./workdir/${RUN_NAME}"