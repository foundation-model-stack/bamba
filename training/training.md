Here are the setup details and command on how you can train the model:

``` python
git clone https://github.com/foundation-model-stack/fms-fsdp.git
cd fms-fsdp && pip install -e .

torchrun --nnodes=24 --node_rank=0 --nproc_per_node=8 \
    main_training_mamba.py \
      --model_variant=mamba_9.8b \
      --tokenizer_path="/path/to/tokenizer/" \
      --data_path="/path/to/datasets/" \
      --datasets="subdataset1,subdataset2,subdataset3,subdataset4,..." \
      --weights="1,1,1,1,.." \
      --seq_length=4096 \
      --vocab_size=128256 \
      --logical_shards=960 \
      --ckpt_load_path="/path/to/model/checkpoint" \
      --ckpt_save_path="/path/to/save/model/checkpoint" \
      --sharding_strategy="fsdp" \
      --batch_size=2 \
      --learning_rate=3e-4 \
      --num_steps=1280000 \
      --report_interval=100 \
      --checkpoint_interval=20000 \
```
