

# Number of GPUs per GPU worker
export GPUS_PER_NODE=8
# Number of GPU workers, for single-worker training, please set to 1
export NUM_NODES=$SLURM_NNODES
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# The port for communication
export MASTER_PORT=12350
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=$SLURM_NODEID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "RANK :$RANK"
echo "NUM_NODES :$NUM_NODES"
echo "GPUS_PER_NODE :$GPUS_PER_NODE"

export MIOPEN_USER_DB_PATH=/lus/home/NAT/gda2204/mshukor/.config/miopen_${MASTER_ADDR}_${SLURM_PROCID}/

echo "MIOPEN_USER_DB_PATH :$MIOPEN_USER_DB_PATH"

num_workers=0


exp_name=unival_audio_caption_s2_onlylinear

ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs

save_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
save_dir=${save_base_log_dir}/ofa/checkpoints/caption/${exp_name}
log_dir=${save_dir}

mkdir -p $log_dir $save_dir

bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module



image_dir=${base_data_dir}


data_dir=${base_data_dir}/ofa/audio_data/caption_data
data=${data_dir}/audiocaps_train_1.tsv,${data_dir}/audiocaps_train_2.tsv,${data_dir}/audiocaps_train_3.tsv,${data_dir}/audiocaps_train_4.tsv,${data_dir}/audiocaps_train_5.tsv,${data_dir}/audiocaps_train_6.tsv,${data_dir}/audiocaps_train_7.tsv,${data_dir}/audiocaps_train_8.tsv,${data_dir}/audiocaps_train_9.tsv,${data_dir}/audiocaps_train_10.tsv,${data_dir}/audiocaps_test.tsv
eval_cider_cached=${data_dir}/cider_cached_tokens/audiocaps-test-words.p
  

restore_file=${base_log_dir}/ofa/checkpoints/pretrain/unival_s2_hs/checkpoint3.pt



selected_cols=0,4,2

task=audio_caption
arch=unival_base
pretrained_model=  


criterion=adjust_label_smoothed_encouraging_loss
label_smoothing=0.1
lr=5e-4
max_epoch=25
warmup_ratio=0.06
batch_size=8
update_freq=2
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
# patch_image_size=480
drop_worst_ratio=0.2




###
image_encoder_name=timm_resnet #vit_base_patch16_224 144
patch_image_size=384
resnet_type=resnet101

resnet_model_path=${base_log_dir}/pretrained_models/resnet101_a1h-36d3f2aa.pth

# video
video_encoder_name=all_resnext101 # 49 
patch_frame_size=224
video_model_path=${base_log_dir}/pretrained_models/3dcnn/resnext-101-kinetics.pth #${base_log_dir}/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth
num_frames=8

# audio 
audio_encoder_name=pann_cnn14
audio_model_path=${base_log_dir}/pretrained_models/Cnn14_mAP_0.431.pth 
mel_bins=64
hop_size=200 # 155 tok



save_interval=1
validate_interval_updates=2000
save_interval_updates=0


sample_patch_num='--sample-patch-num=784' # ''

eval_args='--eval-args={"beam":5,"unnormalized":true,"temperature":1.0,"stop_on_max_len":true}'


drop_worst_ratio=0.05 # modified from 0.2 for el
log_end=0.75  # for el
drop_best_ratio=0.05
drop_best_after=6000
drop_worst_after=6000

echo "max_epoch "${max_epoch}
for warmup_ratio in {0.06,}; do
  echo "warmup_ratio "${warmup_ratio}
  for drop_worst_after in {6000,}; do
    echo "drop_worst_after "${drop_worst_after}

    log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}".log"
    save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}
    mkdir -p $save_path

    python3 -m torch.distributed.launch \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port=${MASTER_PORT} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} \
        --use_env ${ofa_dir}/train.py \
        $data \
        --selected-cols=${selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --save-dir=${save_path} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --log-format=simple --log-interval=10 \
        --fixed-validation-seed=7 \
        --no-epoch-checkpoints --keep-best-checkpoints=1 \
        --save-interval=${save_interval} --validate-interval=1 \
        --save-interval-updates=${save_interval_updates} --validate-interval-updates=${validate_interval_updates} \
        --eval-cider \
        --eval-cider-cached-tokens=${eval_cider_cached} \
        --eval-args='{"beam":5,"max_len_b":16,"no_repeat_ngram_size":3}' \
        --best-checkpoint-metric=cider --maximize-best-checkpoint-metric \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --find-unused-parameters \
        --freeze-encoder-embedding \
        --freeze-decoder-embedding \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --disable-entangle \
        --num-bins=${num_bins} \
        --patch-image-size=${patch_image_size} \
        --drop-worst-ratio=${drop_worst_ratio} \
        --drop-worst-after=${drop_worst_after} \
        --fp16-scale-window=512 \
        --fp16 \
        --num-workers=0 \
        --image-encoder-name=${image_encoder_name} \
        --image-dir=${image_dir} \
        --video-encoder-name=${video_encoder_name} \
        --video-model-path=${video_model_path} \
        --patch-frame-size=${patch_frame_size} \
        ${sample_patch_num} \
        ${eval_args} \
        --num-frames=${num_frames} \
        --resnet-type=${resnet_type} \
        --resnet-model-path=${resnet_model_path} \
        --audio-encoder-name=${audio_encoder_name} \
        --audio-model-path=${audio_model_path} \
        --mel-bins=${mel_bins} \
        --hop-size=${hop_size} \
        --reset-dataloader --reset-meters --reset-optimizer \
        --save-on-cuda \
        --strict \
        --freeze-encoder \
        --freeze-decoder \
        --freeze-audio-encoder \
        --audio-sample-rate \
        --strict \
        --log-end ${log_end} --drop-best-ratio ${drop_best_ratio} --drop-best-after ${drop_best_after} 

  done
done
