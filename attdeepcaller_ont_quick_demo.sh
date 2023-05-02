PLATFORM='ont'

## HG003
INPUT_DIR="/work/Clair3-main/data/datatest/HG003/GRCh38/"
OUTPUT_DIR="/work/Clair3-main-sy/data/mytrainmodel/test/hg003-0305"
##mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
THREADS=200
REF="GRCh38.fa"
BAM="HG003_GRCh38.bam"
BASELINE_VCF_FILE_PATH="HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
BASELINE_BED_FILE_PATH="HG003_GRCh38.bed"
OUTPUT_VCF_FILE_PATH="merge_output.vcf.gz"

./run_attdeepcaller.sh \
  --bam_fn=${INPUT_DIR}/${BAM} \
  --ref_fn=${INPUT_DIR}/${REF} \
  --threads=${THREADS} \
  --platform=${PLATFORM} \
  --model_path=/work/Clair3-main-sy/data/mytrainmodel/ont/ \
  --output=${OUTPUT_DIR}