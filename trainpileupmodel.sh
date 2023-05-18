#### 1. Setup variables
#```bash
# Setup executable variables
ATTDEEPCALLER="/work/Attdeepcaller/attdeepcaller.py"
PYPY="/home/user/anaconda3/envs/attdeepcaller/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/attdeepcaller/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/attdeepcaller/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/attdeepcaller/bin/python3"                             # e.g. python3

# Input parameters
PLATFORM="ont"                         # e.g. {ont, hifi, ilmn}
UNIFIED_VCF_FILE_PATH="/work/attdeepcaller/data/outputref-HG001-GRCh38/unified.vcf.gz"       # e.g. hg002.unified.vcf.gz
ALL_BAM_FILE_PATH="/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.bam"           # e.g. hg002.bam
DEPTHS="1000"                  # e.g. 1000 (means no subsample)
ALL_REFERENCE_FILE_PATH="/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.fa"   # e.g. hg002.fasta
ALL_BED_FILE_PATH="/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.bed"           # e.g. hg002.bed
ALL_SAMPLE="GRCh38"                    # e.g. hg002
OUTPUT_DIR="/work/attdeepcaller/data/HG001-600-1000-trainoutput/pileup/modeltest"                   # e.g. output_folder
# Each line represent one input BAM with a matched coverage in the "DEPTH" array
## check the "Training data subsamping" section on how to apply BAM subsampling
ALL_BAM_FILE_PATH=(
'/work/attdeepcaller/data/datatest/subsample-HG001/GRCh38_1000.bam'
'/work/attdeepcaller/data/datatest/subsample-HG001/600_GRCh38.bam'
'/work/attdeepcaller/data/datatest/subsample-HG002/GRCh38_1000.bam'
)

# Each line represents subsample ration to each sample, 1000 if no subsampling applies
DEPTHS=(
1000
600
1000
)

# Each line represents one input sample name
ALL_SAMPLE=(
'hg001'
'hg001'
'hg002'
)

# Each line represents the reference file of each sample
ALL_REFERENCE_FILE_PATH=(
'/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.fa'
'/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.fa'
'/work/attdeepcaller/data/datatest/HG002/GRCh38/GRCh38.fa'
)

# Each line represents one BED region file for each sample
ALL_BED_FILE_PATH=(
'/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.bed'
'/work/attdeepcaller/data/datatest/HG001/GRCh38/GRCh38.bed'
'/work/attdeepcaller/data/datatest/HG002/GRCh38/HG002_GRCh38.bed'
)

# Each line represents one representation-unified VCF file for each sample
UNIFIED_VCF_FILE_PATH=(
'/work/attdeepcaller/data/outputref-HG001_GRCh38/unified.vcf.gz'
'/work/attdeepcaller/data/outputref-HG001-GRCh38-600/unified.vcf.gz'
'/work/attdeepcaller/data/outputref-HG002_GRCh38/unified.vcf.gz'
)

# Chromosome prefix ("chr" if chromosome names have the "chr" prefix)
CHR_PREFIX="chr"

# array of chromosomes (do not include tge "chr" prefix) to train in all sample
## pls note that in the pretrained ATTDEEPCALLER models, we have excluded chr20 as a hold-out set.
CHR=(21 22)

# Number of threads to be used
THREADS=8
THREADS_LOW=$((${THREADS}*3/4))
if [[ ${THREADS_LOW} < 1 ]]; then THREADS_LOW=1; fi

# Number of chucks to be divided into for parallel processing
chunk_num=15
CHUNK_LIST=`seq 1 ${chunk_num}`

# The number of chucks to be divided for bin file generation for parallel processing
bin_chunk_num=1
BIN_CHUNK_LIST=`seq 1 ${bin_chunk_num}`

# Minimum SNP and INDEL AF required for a candidate variant
MIN_SNP_AF=0.08
MIN_INDEL_AF=0.15

# Maximum non-variant ratio for pileup model training, for pileup model training, we use variant:non-variant = 1:5
MAXIMUM_NON_VARIANT_RATIO=5

#```

#### 2. Create temporary working folders for each submodule
#```bash
# Temporary working directories
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
TENSOR_CANDIDATE_PATH="${DATASET_FOLDER_PATH}/tensor_can"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"
SPLIT_BED_PATH="${DATASET_FOLDER_PATH}/split_beds"
VAR_OUTPUT_PATH="${DATASET_FOLDER_PATH}/var"

mkdir -p ${DATASET_FOLDER_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH}
mkdir -p ${BINS_FOLDER_PATH}
mkdir -p ${SPLIT_BED_PATH}
mkdir -p ${VAR_OUTPUT_PATH}

#```

#### 3. Split and extend bed regions using the `SplitExtendBed` submodule
#```bash
cd ${OUTPUT_DIR}

# Split BED file regions according to the contig names and extend the bed regions
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${ATTDEEPCALLER} SplitExtendBed \
    --bed_fn {4} \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]}

#```

##### 4. Get truth variants from unified VCF using the `GetTruth` submodule
#
##```bash
${PARALLEL} --joblog ${VAR_OUTPUT_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${ATTDEEPCALLER} GetTruth \
    --vcf_fn {4} \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${UNIFIED_VCF_FILE_PATH[@]}
#
##```
#
##### 5. Create pileup tensor using the `CreateTrainingTensor` submodule
#
##```bash
## Create pileup tensor for model training
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_pileup.log -j${THREADS_LOW} \
"${PYPY} ${ATTDEEPCALLER} CreateTrainingTensor \
    --bam_fn {4} \
    --ref_fn {5} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_{7} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --snp_min_af ${MIN_SNP_AF} \
    --indel_min_af ${MIN_INDEL_AF} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --bed_fn {6} \
    --pileup \
    --platform ${PLATFORM} \
    --shuffle \
    --maximum_non_variant_ratio ${MAXIMUM_NON_VARIANT_RATIO} \
    --chunk_id {7} \
    --chunk_num ${chunk_num}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} ::: ${CHUNK_LIST[@]} |& tee  ${DATASET_FOLDER_PATH}/CTP.log
#
##```
#
##**Options**
##
## - `--allow_duplicate_chr_pos` : for multiple coverages training, this option is required to to allow different coverage training samples at the same variant site.
## - `--shuffle` :  as the input tensors are created in the order of starting positions, this option shuffles the training data in each chunk. During the training process, we also apply index reshuffling in each epoch.
## - `--maximum_non_variant_ratio` :  we set a maximum non-variant ratio (variant:non-variant = 1:5) for pileup model training, non-variants are randomly selected from the candidate set if the ratio is exceeded, or all non-variants will be used for training otherwise.
## - `--max_depth` :  set the depth cap of every genome position. Pileup input summarizes position-level read alignments where depth information varies in the training materials. If not contrained by computational resources, we recommend setting the depth cap to the maximum depth coverage of the training materials.
#
##### 6. Merge compressed binaries using the `MergeBin` submodule
#
##```bash
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${ATTDEEPCALLER} MergeBin \
    ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_* \
    --out_fn ${BINS_FOLDER_PATH}/bin_{2}_{3}_{1} \
    --pileup" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
##```
##
##----