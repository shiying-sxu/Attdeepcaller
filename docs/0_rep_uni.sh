# Setup variables
CLAIR3="/work/attdeepcaller/attdeepcaller.py"                                  # clair3.py
PYPY="/home/user/anaconda3/envs/attdeepcaller/bin/pypy3.6"                                 # e.g. pypy3
PARALLEL="/home/user/anaconda3/envs/attdeepcaller/bin/parallel"                         # e.g. parallel
SAMTOOLS="/home/user/anaconda3/envs/attdeepcaller/bin/samtools"                         # e.g. samtools
PYTHON3="/home/user/anaconda3/envs/attdeepcaller/bin/python3"                             # e.g. python3
WHATSHAP="/home/user/anaconda3/envs/attdeepcaller/bin/whatshap"                         # e.g. whatshap
TABIX="/home/user/anaconda3/envs/attdeepcaller/bin/tabix"                             # e.g. tabix
# Input parameters
PLATFORM="ont"                       # e.g. {ont, hifi, ilmn}



#HG002_chr20
VCF_FILE_PATH="/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG002_GRCh38_20_v4.2.1_benchmark.vcf.gz"   # [YOUR_VCF_FILE_PATH]e.g. hg003.vcf.gz
BAM_FILE_PATH="/data2/GUPPY5/chr20/hg002_chr20.bam"     # e.g. hg003.bam   alignment.bam
REFERENCE_FILE_PATH="/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/GRCh38_no_alt_chr20.fa"   # e.g. hg003.fasta
BED_FILE_PATH="/work/Clair3-Trio-trio/data/clair3_trio_quick_demo/HG002_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed"    # e.g. hg003.bed
OUTPUT_DIR="/data2/GUPPY5/OUTPUT_UNI/chr20/HG002"					       # e.g. output

#HG003
#VCF_FILE_PATH="/data2/GUPPY5/chr22/HG003_chr22.vcf.gz"   # [YOUR_VCF_FILE_PATH]e.g. hg003.vcf.gz
#BAM_FILE_PATH="/data2/GUPPY5/HG003/hg003_chr22.bam"     # e.g. hg003.bam   alignment.bam
#REFERENCE_FILE_PATH="/data2/GUPPY5/chr22/chr22.fa"   # e.g. hg003.fasta
#BED_FILE_PATH="/data2/GUPPY5/chr22/HG003_chr22.bed"    # e.g. hg003.bed
#OUTPUT_DIR="/data2/GUPPY5/OUTPUT_UNI/HG003"					       # e.g. output
###HG004
#VCF_FILE_PATH="/data2/GUPPY5/chr22/HG004_chr22.vcf.gz"   # [YOUR_VCF_FILE_PATH]e.g. hg003.vcf.gz
#BAM_FILE_PATH="/data2/GUPPY5/HG004/hg004_chr22.bam"     # e.g. hg003.bam   alignment.bam
#REFERENCE_FILE_PATH="/data2/GUPPY5/chr22/chr22.fa"   # e.g. hg003.fasta
#BED_FILE_PATH="/data2/GUPPY5/chr22/HG004_chr22.bed"    # e.g. hg003.bed
#OUTPUT_DIR="/data2/GUPPY5/OUTPUT_UNI/HG004"					       # e.g. output

# Chromosome prefix ("chr" if chromosome names have the "chr" prefix)染色体前缀
CHR_PREFIX="chr"

# array of chromosomes (do not include "chr"-prefix)染色体阵列
CHR=(20)

# Number of threads to be used
THREADS=24

# The number of chucks to be divided into for parallel processing
#并行加工需分割的卡盘数量
chunk_num=15
CHUNK_LIST=`seq 1 ${chunk_num}`

# Minimum AF required for a candidate variant
MIN_AF=0.08    #0.08

# Temporary working directory
SPLIT_BED_PATH="${OUTPUT_DIR}/split_beds"
VCF_OUTPUT_PATH="${OUTPUT_DIR}/vcf_output"
VAR_OUTPUT_PATH="${OUTPUT_DIR}/var"
PHASE_VCF_PATH="${OUTPUT_DIR}/phased_vcf"
PHASE_BAM_PATH="${OUTPUT_DIR}/phased_bam"

mkdir -p ${SPLIT_BED_PATH}
mkdir -p ${VCF_OUTPUT_PATH}
mkdir -p ${VAR_OUTPUT_PATH}
mkdir -p ${PHASE_VCF_PATH}
mkdir -p ${PHASE_BAM_PATH}

#### 2.  Phase VCF file using WhatsHap

# To apply representation unification,  using a phased read alignment is highly recommended in order to get more precious unified result.

# ```bash
cd ${OUTPUT_DIR}

# WhatsHap phasing vcf file if vcf file includes '|' in INFO tag
${WHATSHAP} unphase ${VCF_FILE_PATH} > ${OUTPUT_DIR}/INPUT.vcf.gz

# WhatsHap phase vcf file
${PARALLEL} --joblog ${PHASE_VCF_PATH}/phase.log -j${THREADS} \
"${WHATSHAP} phase \
    --output ${PHASE_VCF_PATH}/phased_{1}.vcf.gz \
    --reference ${REFERENCE_FILE_PATH} \
    --chromosome ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    --distrust-genotypes \
    ${OUTPUT_DIR}/INPUT.vcf.gz \
    ${BAM_FILE_PATH}" ::: ${CHR[@]}

# Index phased vcf file
${PARALLEL} -j ${THREADS} tabix -p vcf ${PHASE_VCF_PATH}/phased_{1}.vcf.gz ::: ${CHR[@]}
# ```

#### 3.  Haplotag read alignment using WhatsHap

# ```bash
# WhatsHap haplotags bam file
${PARALLEL} --joblog ${PHASE_BAM_PATH}/haplotag.log -j${THREADS} \
"${WHATSHAP} haplotag \
    --output ${PHASE_BAM_PATH}/{1}.bam \
    --reference ${REFERENCE_FILE_PATH} \
    --regions ${CHR_PREFIX}{1} \
    --ignore-read-groups \
    ${PHASE_VCF_PATH}/phased_{1}.vcf.gz \
    ${BAM_FILE_PATH}" ::: ${CHR[@]}

# Index the phased bam file using samtools
${PARALLEL} --joblog ${PHASE_BAM_PATH}/index.log -j ${THREADS} ${SAMTOOLS} index -@12 ${PHASE_BAM_PATH}/{1}.bam ::: ${CHR[@]}

# ```

#### 4.  Prepare true variant set and candidate input

# ```bash
# Split bed file regions according to the contig name and extend bed region
${PARALLEL} --joblog ${SPLIT_BED_PATH}/split_extend_bed.log -j${THREADS} \
"${PYPY} ${CLAIR3} SplitExtendBed \
    --bed_fn ${BED_FILE_PATH} \
    --output_fn ${SPLIT_BED_PATH}/{1} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]}

#Get true variant label information from VCF file
${PARALLEL} --joblog ${VAR_OUTPUT_PATH}/get_truth.log -j${THREADS} \
"${PYPY} ${CLAIR3} GetTruth \
    --vcf_fn ${PHASE_VCF_PATH}/phased_{1}.vcf.gz \
    --ctgName ${CHR_PREFIX}{1} \
    --var_fn ${VAR_OUTPUT_PATH}/var_{1}" ::: ${CHR[@]}

CANDIDATE_DETAILS_PATH="${OUTPUT_DIR}/candidate_details"
mkdir -p ${CANDIDATE_DETAILS_PATH}
# Create candidate details for representation unification
${PARALLEL} --joblog ${CANDIDATE_DETAILS_PATH}/create_tensor.log -j${THREADS} \
"${PYPY} ${CLAIR3} CreateTensorFullAlignment \
    --bam_fn ${PHASE_BAM_PATH}/{1}.bam \
    --ref_fn ${REFERENCE_FILE_PATH} \
    --indel_fn ${CANDIDATE_DETAILS_PATH}/{1}_{2} \
    --ctgName ${CHR_PREFIX}{1} \
    --samtools ${SAMTOOLS} \
    --min_af ${MIN_AF} \
    --extend_bed ${SPLIT_BED_PATH}/{1} \
    --unify_repre_fn ${CANDIDATE_DETAILS_PATH}/{1}_{2} \
    --unify_repre \
    --phasing_info_in_bam \
    --bed_fn ${BED_FILE_PATH} \
    --chunk_id {2} \
    --chunk_num ${chunk_num}" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]}
# ```

#### 5.  Unify Representation for true variant set and candidate sites
#真实变体集和候选位点的统一表示
# ```bash
${PARALLEL} --joblog ${OUTPUT_DIR}/unify_repre.log -j${THREADS} \
"${PYPY} ${CLAIR3} UnifyRepresentation \
    --bam_fn ${PHASE_BAM_PATH}/{1}.bam \
    --var_fn ${VAR_OUTPUT_PATH}/var_{1} \
    --ref_fn ${REFERENCE_FILE_PATH} \
    --bed_fn ${BED_FILE_PATH} \
    --extend_bed ${SPLIT_BED_PATH}/{1} \
    --output_vcf_fn ${VCF_OUTPUT_PATH}/vcf_{1}_{2} \
    --min_af ${MIN_AF} \
    --chunk_id {2} \
    --chunk_num ${chunk_num} \
    --platform ${PLATFORM} \
    --ctgName ${CHR_PREFIX}{1}" ::: ${CHR[@]} ::: ${CHUNK_LIST[@]} > ${OUTPUT_DIR}/RU.log


# ```

#### 6.  Merge and sort unified VCF output
#合并和排序统一的 VCF 输出

# ```bash
cat ${VCF_OUTPUT_PATH}/vcf_* | ${PYPY} ${CLAIR3} SortVcf --output_fn ${OUTPUT_DIR}/unified.vcf
bgzip -f ${OUTPUT_DIR}/unified.vcf
tabix -f -p vcf ${OUTPUT_DIR}/unified.vcf.gz

# ```

#### 7.  Benchmark using unified VCF and true variant set (optional)
#使用统一 VCF 和真实变体集的基准测试（可选）
# ```bash
# Install hap.py if not installed
# conda config --add channels defaults
# conda config --add channels bioconda
# conda config --add channels conda-forge
# conda create -n happy-env -c bioconda hap.py -y
# conda install -c bioconda rtg-tools -y
# conda activate happy-env
#
# # Benchmark using hap.py
# hap.py \
#     ${VCF_FILE_PATH} \
#     ${OUTPUT_DIR}/unified.vcf.gz \
#     -o ${OUTPUT_DIR}/happy \
#     -r ${REFERENCE_FILE_PATH} \
#     -f ${BED_FILE_PATH} \
#     --threads ${THREADS} \
#     --engine=vcfeval \
#     -l "[YOUR_BENCHMARK_REGION]" # e.g. chr22

# ```
