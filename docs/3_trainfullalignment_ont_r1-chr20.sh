# Train a model for ATTDEEPCALLER full-alignment calling (revision 1)


## I. Training data phasing and haplotaging


#### 1. Setup variables

# Setup executable variables
ATTDEEPCALLER="/work/attdeepcaller/attdeepcaller.py"                                  # clair3.py
PYPY="/home/user/anaconda3/envs/attdeepcaller/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/attdeepcaller/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/attdeepcaller/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/attdeepcaller/bin/python3"                             # e.g. python3
WHATSHAP="/home/user/anaconda3/envs/attdeepcaller/bin/whatshap"                         # e.g. whatshap
TABIX="/home/user/anaconda3/envs/attdeepcaller/bin/tabix"                             # e.g. tabix


ATTDEEPCALLER_PATH="/home/user/anaconda3/envs/atttrio/bin"                    # ATTDEEPCALLER installation path






# Input parameters
PLATFORM="ont"                     # e.g. {ont, hifi, ilmn}
OUTPUT_DIR="/data2/attdeepcaller/data/GUPPY5-trainoutput/fullalignment/modeltest-train_chr20-ATT"               # e.g. output



ALL_UNPHASED_BAM_FILE_PATH=(
'/data2/GUPPY5/chr20/hg002_chr20.bam'
'/data2/GUPPY5/chr20/hg003_chr20.bam'
'/data2/GUPPY5/chr20/hg004_chr20.bam'
)
# Each line represents a sample, a sample can be specified multiple times to allow downsampling
ALL_SAMPLE=(
'hg002'
'hg003'
'hg004'
)

# A downsampling numerator (1000 as denominator) for each sample in ALL_SAMPLE, 1000 means no downsampling, 800 means 80% (800/1000)
DEPTHS=(
1000
1000
1000
)

ALL_REFERENCE_FILE_PATH=(
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/GRCh38_no_alt_chr20.fa'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/GRCh38_no_alt_chr20.fa'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/GRCh38_no_alt_chr20.fa'
)



ALL_BED_FILE_PATH=(
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG002_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG003_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG004_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed'
)



TRUTH_VCF_FILE_PATH=(
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG002_GRCh38_20_v4.2.1_benchmark.vcf.gz'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG003_GRCh38_20_v4.2.1_benchmark.vcf.gz'
'/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG004_GRCh38_20_v4.2.1_benchmark.vcf.gz'
)

UNIFIED_VCF_FILE_PATH=(
'/data2/GUPPY5/OUTPUT_UNI/chr20/HG002/unified.vcf.gz'
'/data2/GUPPY5/OUTPUT_UNI/chr20/HG003/unified.vcf.gz'
'/data2/GUPPY5/OUTPUT_UNI/chr20/HG004/unified.vcf.gz'
)
# Chromosome prefix ("chr" if chromosome names have the "chr"-prefix)
CHR_PREFIX="chr"

# array of chromosomes (do not include "chr"-prefix) to training in all sample

CHR=(20)

# Number of threads to be used
THREADS=40
THREADS_LOW=$((${THREADS}*3/4))
if [[ ${THREADS_LOW} < 1 ]]; then THREADS_LOW=1; fi

# Number of chucks to be divided into for parallel processing
chunk_num=15
CHUNK_LIST=`seq 1 ${chunk_num}`

# Maximum non-variant ratio for full-alignment model training, for full-alignment model training, we use variant :non-variant = 1 : 1
MAXIMUM_NON_VARIANT_RATIO=1

# Temporary working directory
DATASET_FOLDER_PATH="${OUTPUT_DIR}/build"
TENSOR_CANDIDATE_PATH="${DATASET_FOLDER_PATH}/tensor_can"
BINS_FOLDER_PATH="${DATASET_FOLDER_PATH}/bins"
CANDIDATE_DETAILS_PATH="${DATASET_FOLDER_PATH}/candidate_details"
CANDIDATE_BED_PATH="${DATASET_FOLDER_PATH}/candidate_bed"
SPLIT_BED_PATH="${DATASET_FOLDER_PATH}/split_beds"
VAR_OUTPUT_PATH="${DATASET_FOLDER_PATH}/var"
PILEUP_OUTPUT_PATH="${OUTPUT_DIR}/pileup_output"
UNPHASED_TRUTH_VCF_PATH="${OUTPUT_DIR}/unphased_truth_vcf"
PHASE_VCF_PATH="${OUTPUT_DIR}/phased_vcf"
PHASE_BAM_PATH="${OUTPUT_DIR}/phased_bam"

mkdir -p ${DATASET_FOLDER_PATH}
mkdir -p ${TENSOR_CANDIDATE_PATH}
mkdir -p ${BINS_FOLDER_PATH}
mkdir -p ${CANDIDATE_DETAILS_PATH}
mkdir -p ${SPLIT_BED_PATH}
mkdir -p ${VAR_OUTPUT_PATH}
mkdir -p ${CANDIDATE_BED_PATH}
mkdir -p ${PILEUP_OUTPUT_PATH}
mkdir -p ${UNPHASED_TRUTH_VCF_PATH}
mkdir -p ${PHASE_VCF_PATH}
mkdir -p ${PHASE_BAM_PATH}
#```

#### 2.  Phase VCF file using WhatsHap

#```bash
cd ${OUTPUT_DIR}

# Remove the phasing information if the VCF input is already phased
${PARALLEL} -j${THREADS} "${WHATSHAP} unphase {3} > ${UNPHASED_TRUTH_VCF_PATH}/unphased_truth_{1}_{2}.vcf.gz" ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${TRUTH_VCF_FILE_PATH[@]}

# WhatsHap phasing
${PARALLEL} --joblog ${PHASE_VCF_PATH}/phase.log -j${THREADS} \
"${WHATSHAP} phase \
    --output ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz \
    --reference {5} \
    --chromosome ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    --distrust-genotypes \
    ${UNPHASED_TRUTH_VCF_PATH}/unphased_truth_{2}_{3}.vcf.gz \
    {4}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} |& tee ${PHASE_VCF_PATH}/PHASE.log

# Index the phased VCF files using tabix, which is neccesary for read haplotagging
${PARALLEL} -j ${THREADS} ${TABIX} -p vcf ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
#```

#### 3.  Haplotag read alignments using WhatsHap

#```bash
# WhatsHap haplotaging
${PARALLEL} --joblog ${PHASE_BAM_PATH}/haplotag.log -j${THREADS} \
"${WHATSHAP} haplotag \
    --output ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam \
    --reference {5} \
    --regions ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    ${PHASE_VCF_PATH}/phased_{2}_{3}_{1}.vcf.gz \
    {4}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} |& tee ${PHASE_VCF_PATH}/HAPLOTAG.log

# Index the phased bam files using samtools
${PARALLEL} --joblog ${PHASE_BAM_PATH}/index.log -j ${THREADS} ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}


## II. Build compressed binary files for full-alignment model training

#This section shows how to build multiple compressed tensor binary file for multiple samples with multiple coverages.

#### 1. Run attdeepcaller pileup model

#```bash
# Call variants using attdeepcaller‘s pileup model with the --pileup_only option
# Only select the candidates in the high-confident BED regions for model training (with --bed_fn)
${PARALLEL} -j1 /work/attdeepcaller/run_attdeepcaller.sh \
  --bam_fn={3} \
  --ref_fn={4} \
  --threads=${THREADS} \
  --platform=${PLATFORM} \
  --model_path=/data2/attdeepcaller/data/model/ATT-chr20/ \
  --output=${PILEUP_OUTPUT_PATH}/{1}_{2} \
  --bed_fn={5} \
  --pileup_only ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]}
#```

#### 2. Select low-quality pileup candidates using the `SelectHetSnp` submodule

#```bash
# Select all pileup called variants (0/1, 1/1 and 1/2) and some pileup reference calls (0/0) for full-alignment model training
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/select_pileup_candidates.log -j${THREADS} \
"${PYPY} ${ATTDEEPCALLER} SelectHetSnp \
--alt_fn ${PILEUP_OUTPUT_PATH}/{2}_{3}/pileup.vcf.gz \
--split_folder ${CANDIDATE_BED_PATH} \
--sampleName {2} \
--depth {3} \
--ref_pct_full 0.15 \
--var_pct_full 1.0 \
--chunk_num ${chunk_num} \
--phasing_info_in_bam \
--phase \
--ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
#```

#### 3. Split and extend bed regions using the `SplitExtendBed` submodule

#```bash
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${ATTDEEPCALLER} SplitExtendBed \
    --bed_fn {4} \
    --output_fn ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_BED_FILE_PATH[@]}
#```

#### 4. Get truth variants from unified VCF using the `GetTruth` submodule

#```bash
# Convert an unified VCF file into a simplified var file
${PARALLEL} --joblog ${VAR_OUTPUT_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${ATTDEEPCALLER} GetTruth \
    --vcf_fn {4} \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${UNIFIED_VCF_FILE_PATH[@]}
#```

#### 5. Create full-alignment tensor using the `CreateTrainingTensor` submodule

#```bash
# Create full-alignment tensors for model training
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/create_tensor_full_alignment.log -j${THREADS_LOW} \
"${PYPY} ${ATTDEEPCALLER} CreateTrainingTensor \
    --bam_fn ${PHASE_BAM_PATH}/{2}_{3}_{1}.bam \
    --ref_fn {5} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{2}_{3}_{1} \
    --bin_fn ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_{7} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --extend_bed ${SPLIT_BED_PATH}/{2}_{3}_{1} \
    --full_aln_regions ${CANDIDATE_BED_PATH}/{2}_{3}_{1}_{7} \
    --bed_fn {6} \
    --phasing_info_in_bam \
    --add_no_phasing_data_training \
    --allow_duplicate_chr_pos \
    --platform ${PLATFORM} \
    --shuffle \
    --maximum_non_variant_ratio ${MAXIMUM_NON_VARIANT_RATIO} \
    --chunk_id {7} \
    --chunk_num ${chunk_num}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]} :::+ ${ALL_UNPHASED_BAM_FILE_PATH[@]} :::+ ${ALL_REFERENCE_FILE_PATH[@]} :::+ ${ALL_BED_FILE_PATH[@]} ::: ${CHUNK_LIST[@]}
#```

#**Options**
#
# - `--phasing_info_in_bam` : enabled by default, indicating the input BAM is phased using WhatsHap's `haplotag` module, and phased alignments are having a `HP` tag with haplotype details.
# - `--allow_duplicate_chr_pos` : for multiple coverages training, this option is required to to allow different coverage training samples at the same variant site.
# - `--shuffle` :  as the input tensors are created in the order of starting positions, this option shuffles the training data in each chunk. During the training process, we also apply index reshuffling in each epoch.
# - `--maximum_non_variant_ratio` :  we set a maximum non-variant ratio (variant:non-variant = 1:1) for full-alignment model training, non-variants are randomly selected from the candidate set if the ratio is exceeded, or all non-variants will be used for training otherwise.
# - `--add_no_phasing_data_training` : also include unphased alignments in additional to the phased alignments for training. We found including unphased alignments increased model robustness.
# - `--full_aln_regions` : provide the pileup candidate regions to be included in full-alignment based calling.

#### 6. Merge compressed binaries using the `MergeBin` submodule

#```bash
# Merge compressed binaries
${PARALLEL} --joblog ${DATASET_FOLDER_PATH}/mergeBin.log -j${THREADS} \
"${PYTHON3} ${ATTDEEPCALLER} MergeBin \
    ${TENSOR_CANDIDATE_PATH}/tensor_{2}_{3}_{1}_* \
    --platform ${PLATFORM} \
    --out_fn ${BINS_FOLDER_PATH}/bin_{2}_{3}_{1}" ::: ${CHR[@]} ::: ${ALL_SAMPLE[@]} :::+ ${DEPTHS[@]}
#```
#
#----
