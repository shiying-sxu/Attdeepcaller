CLAIR3="/work/attdeepcaller/attdeepcaller.py"                                   # attdeepcaller.py
PYPY="/home/user/anaconda3/envs/atttrio/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/atttrio/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/atttrio/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/atttrio/bin/python3"                             # e.g. python3




# Input parameters
PLATFORM="ont"                         # e.g. {ont, hifi, ilmn}





OUTPUT_DIR="/data2/attdeepcaller/data/GUPPY5-trainoutput/fullalignment/modeltest-train_chr20-ATT"                # e.g. output
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"
MODEL_FOLDER_PATH="${OUTPUT_DIR}/train-fullalign-resnext-CBAM-bilstm"
mkdir -p ${MODEL_FOLDER_PATH}

cd ${MODEL_FOLDER_PATH}

# A single GPU is used for model training
export CUDA_VISIBLE_DEVICES="5"
${PYTHON3} ${CLAIR3} Train \
    --bin_fn ${BINS_FOLDER_PATH} \
    --ochk_prefix ${MODEL_FOLDER_PATH}/full_alignment \
    --add_indel_length True \
    --random_validation \
    --platform ${PLATFORM}