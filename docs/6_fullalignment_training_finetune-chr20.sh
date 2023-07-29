#### 2. full-alignment model fine-tune using pre-trained model (optional)


# Full-alignment model fine-tuning using a new sample
ATTDEEPCALLER="/work/attdeepcaller/attdeepcaller.py"                                   # attdeepcaller.py
PYPY="/home/user/anaconda3/envs/atttrio/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/atttrio/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/atttrio/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/atttrio/bin/python3"                             # e.g. python3
PLATFORM="ont"                         # e.g. {ont, hifi, ilmn}

# Pileup model fine-tuning using a new sample
OUTPUT_DIR="/data2/attdeepcaller/data/GUPPY5-trainoutput/fullalignment/modeltest-train_chr20-ATT"
PRETRAINED_MODEL="/data2/attdeepcaller/data/GUPPY5-trainoutput/fullalignment/modeltest-train_chr20-ATT/train-fullalign-resnext-CBAM-bilstm-39-0.1428/full_alignment.39"
TRAIN_N="finetune0724"
MODEL_FOLDER_PATH="${OUTPUT_DIR}/ATT-chr20/finetune/${TRAIN_N}"
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"
mkdir -p ${MODEL_FOLDER_PATH}

cd ${MODEL_FOLDER_PATH}

export CUDA_VISIBLE_DEVICES="5"
${PYTHON3} ${ATTDEEPCALLER} Train \
    --bin_fn ${BINS_FOLDER_PATH} \
    --ochk_prefix ${MODEL_FOLDER_PATH}/full_alignment \
    --add_indel_length True \
    --random_validation \
    --platform ${PLATFORM} \
    --learning_rate 0.0001 \
    --chkpnt_fn ${PRETRAINED_MODEL}  ##use pre-trained full-alignment model here
