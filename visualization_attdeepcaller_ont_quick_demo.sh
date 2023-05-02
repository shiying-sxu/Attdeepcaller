#**Run hap.py without root privileges for benchmarking (optional)**
#
#```bash
#conda config --add channels defaults
#conda config --add channels bioconda
#conda config --add channels conda-forge
#conda create -n happy-env -c bioconda hap.py -y
#conda install -c bioconda rtg-tools -y
#conda activate happy-env
# sh visualization_attdeepcaller_ont_quick_demo.sh


# Benchmark using hap.py
PLATFORM='ont'
INPUT_DIR="/work/Clair3-main-sy/clair3_ont_quickDemo"
OUTPUT_DIR="${INPUT_DIR}/output/NEWATT"


REF="GRCh38_no_alt_chr20.fa"
BAM="HG003_chr20_demo.bam"
BASELINE_VCF_FILE_PATH="HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz"
BASELINE_BED_FILE_PATH="HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed"
OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"


CONTIGS="chr20"
START_POS=100000
END_POS=300000
THREADS=200
BIN_VERSION="v0.1-r4"


hap.py \
    ${INPUT_DIR}/${BASELINE_VCF_FILE_PATH} \
    ${OUTPUT_DIR}/${OUTPUT_VCF_FILE_PATH} \
    -f "${INPUT_DIR}/${BASELINE_BED_FILE_PATH}" \
    -r "${INPUT_DIR}/${REF}" \
    -o "${OUTPUT_DIR}/happy" \
    -l ${CONTIGS}:${START_POS}-${END_POS} \
    --engine=vcfeval \
    --threads="${THREADS}" \
    --pass-only


#python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/ECA/happy.vcf.gz --output_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/ECA/metrics




