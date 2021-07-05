devices=(6 7)
# tgt_domains="atis AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent"
# tgt_domains="atis PlayMusic"
tgt_domains="PlayMusic"
n_samples=(0)
batch_size=64
workers=8
max_epochs=60
early_stop=10
lr=1e-5
model='BERTPretrainedMRC'
query_types='desp trans' #'desp' #'example' #'trans'
top_n=10
parameter_name=$tgt_domains

device=0
for query_type in ${query_types[@]}
do
    for para in ${parameter_name[@]}
    do
        tgt_domain=$para
        for n in ${n_samples[@]}
        do
            OUTPUT_DIR="train_logs/snips1/${tgt_domain}/${n}/${lr}/"
            mkdir -p $OUTPUT_DIR
            LOG_DIR="logs/snips/${tgt_domain}/${n}/"
            mkdir -p $LOG_DIR
            CUDA_VISIBLE_DEVICES=${devices[$device]} nohup python -u trainer.py --gpus 1 \
            --tgt_domain $tgt_domain --n_samples $n --batch_size $batch_size --workers $workers \
            --model $model --lr $lr  \
            --progress_bar_refresh_rate 10 \
            --max_epochs $max_epochs --early_stop $early_stop \
            --query_type $query_type \
            --top_n ${top_n} \
            --default_root_dir $OUTPUT_DIR  > $LOG_DIR/2-randmonInit-lr${lr}-top_n${top_n}_${query_type}_new_data_coachF1_${model}_bs${batch_size}_ep${max_epochs}.log &
            device=`expr $device + 1`
            device=`expr $device % ${#devices[@]}`
        done
    done
done

            # --load_pretrainedBERT \
            # --only_test \
