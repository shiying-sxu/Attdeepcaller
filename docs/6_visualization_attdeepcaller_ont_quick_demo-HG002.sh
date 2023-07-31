#**Run hap.py without root privileges for benchmarking (optional)**
#
#```bash
#conda config --add channels defaults
#conda config --add channels bioconda
#conda config --add channels conda-forge
#conda create -n happy-env -c bioconda hap.py -y
#conda install -c bioconda rtg-tools -y
#conda activate happy-env
# sh visualization_clair3_ont_quick_demo.sh


## Benchmark using hap.py
PLATFORM='ont'
#
###### HG002_chr20.bam
#INPUT_DIR="/work/Clair3-Trio-trio/data/clair3_trio_quick_demo"
#OUTPUT_DIR="/data2/attdeepcaller/data/guppy5/chr20/HG002-chr20_ALL-ATT"
#
#THREADS=40
#REF="GRCh38_no_alt_chr20.fa"
#BAM="/data2/GUPPY5/chr20/hg002_chr20.bam"
#BASELINE_VCF_FILE_PATH="HG002_GRCh38_20_v4.2.1_benchmark.vcf.gz"
#BASELINE_BED_FILE_PATH="HG002_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed"
#OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"
#BIN_VERSION="v0.1-r4"
#
#hap.py \
#    ${INPUT_DIR}/${BASELINE_VCF_FILE_PATH} \
#    ${OUTPUT_DIR}/${OUTPUT_VCF_FILE_PATH} \
#    -f "${INPUT_DIR}/${BASELINE_BED_FILE_PATH}" \
#    -r "${INPUT_DIR}/${REF}" \
#    -o "${OUTPUT_DIR}/happy" \
#    --engine=vcfeval \
#    --threads="${THREADS}" \
#    --pass-only

### HG002_chr1.bam
INPUT_DIR="/work/Clair3-Trio-trio/data/GUPPY5_chr1/"
OUTPUT_DIR="/data2/attdeepcaller/data/guppy5/chr1/HG002-chr1_ALL-ATT"

THREADS=40
REF="chr1.fasta"
BAM="/work/Clair3-Trio-trio/data/GUPPY5_chr1/HG002_chr1.bam"
BASELINE_VCF_FILE_PATH="HG002_chr1.vcf.gz"
BASELINE_BED_FILE_PATH="HG002_GRCh38_chr1_v4.2.1_benchmark_noinconsistent.bed"
OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"
BIN_VERSION="v0.1-r4"

hap.py \
    ${INPUT_DIR}/${BASELINE_VCF_FILE_PATH} \
    ${OUTPUT_DIR}/${OUTPUT_VCF_FILE_PATH} \
    -f "${INPUT_DIR}/${BASELINE_BED_FILE_PATH}" \
    -r "${INPUT_DIR}/${REF}" \
    -o "${OUTPUT_DIR}/happy" \
    --engine=vcfeval \
    --threads="${THREADS}" \
    --pass-only

### HG003_chr1.bam
#python GetOverallMetrics.py --happy_vcf_fn=/data2/attdeepcaller/data/guppy5/chr20/HG002-chr20_ALL-ATT/happy.vcf.gz --output_fn=/data2/attdeepcaller/data/guppy5/chr20/HG002-chr20_ALL-ATT/metrics
#python GetOverallMetrics.py --happy_vcf_fn=/data2/attdeepcaller/data/guppy5/chr1/HG002-chr1_ALL-ATT/happy.vcf.gz --output_fn=/data2/attdeepcaller/data/guppy5/chr1/HG002-chr1_ALL-ATT/metrics







