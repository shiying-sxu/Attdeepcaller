PLATFORM='ont'

##### HG002_chr20.bam
#INPUT_DIR="/work/Clair3-Trio-trio/data/clair3_trio_quick_demo"
#OUTPUT_DIR="/data2/attdeepcaller/data/guppy5/chr20/HG002-chr20_ALL-ATT"
#
###mkdir -p ${INPUT_DIR}
##mkdir -p ${OUTPUT_DIR}
#THREADS=40
#REF="GRCh38_no_alt_chr20.fa"
#BAM="/data2/GUPPY5/chr20/hg002_chr20.bam"
#BASELINE_VCF_FILE_PATH="HG002_GRCh38_20_v4.2.1_benchmark.vcf.gz"
#BASELINE_BED_FILE_PATH="HG002_GRCh38_20_v4.2.1_benchmark_noinconsistent.bed"
#OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"
#
#
#./run_attdeepcaller.sh \
#  --bam_fn=${BAM} \
#  --ref_fn=${INPUT_DIR}/${REF} \
#  --threads=${THREADS} \
#  --platform=${PLATFORM} \
#  --model_path=/data2/attdeepcaller/data/model/ATT-chr20/finetun-chr20 \
#  --output=${OUTPUT_DIR}

### HG002_chr1.bam
INPUT_DIR="/work/Clair3-Trio-trio/data/GUPPY5_chr1/"
OUTPUT_DIR="/data2/attdeepcaller/data/guppy5/chr1/HG002-chr1_ALL-ATT"
##mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
THREADS=40
REF="chr1.fasta"
BAM="/work/Clair3-Trio-trio/data/GUPPY5_chr1/HG002_chr1.bam"
BASELINE_VCF_FILE_PATH="HG002_chr1.vcf.gz"
BASELINE_BED_FILE_PATH="HG002_GRCh38_chr1_v4.2.1_benchmark_noinconsistent.bed"
OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"

./run_attdeepcaller.sh \
  --bam_fn=${BAM} \
  --ref_fn=${INPUT_DIR}/${REF} \
  --threads=${THREADS} \
  --platform=${PLATFORM} \
  --model_path=/data2/attdeepcaller/data/model/ATT-chr1 \
  --output=${OUTPUT_DIR}