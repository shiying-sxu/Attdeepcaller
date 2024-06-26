## III. Model training

# We provide two optional training mode:
#
# ​	**Option1**: Train  pileup model using new dataset, in this mode, we will use randomly initialized model weights and train the model until reaches max epochs(30) or converge.
#
# ​    **Option2**: Fine-tune pileup model using pre-trained parameters and choose a smaller learning rate for better converge in new dataset.
#
# ***We recommend using the fine-tune mode (option 2) for better robustness.***
#
# #### 1. Pileup model training (option 1)
#
# ```bash

ATTDEEPCALLER="/work/attdeepcaller/attdeepcaller.py"                                   # attdeepcaller.py
PYPY="/home/user/anaconda3/envs/atttrio/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/atttrio/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/atttrio/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/atttrio/bin/python3"                             # e.g. python3

# Input parameters
PLATFORM="ont"                         # e.g. {ont, hifi, ilmn}

OUTPUT_DIR="/data2/attdeepcaller/data/GUPPY5-trainoutput/pileup/modeltest-chr20"
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"

#
MODEL_FOLDER_PATH="${OUTPUT_DIR}/ATT-0721"
mkdir -p ${MODEL_FOLDER_PATH}

cd ${MODEL_FOLDER_PATH}

# A single GPU is used for model training
export CUDA_VISIBLE_DEVICES="3"

${PYTHON3} ${ATTDEEPCALLER} Train \
    --bin_fn ${BINS_FOLDER_PATH} \
    --ochk_prefix ${MODEL_FOLDER_PATH}/pileup \
    --pileup \
    --add_indel_length False \
    --random_validation \
    --platform ${PLATFORM}
