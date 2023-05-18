# Attdeepcaller

# create and activate an environment named attdeepcaller
conda create -n attdeepcaller python=3.9.0 -y
source activate attdeepcaller
# install pypy and packages in the environemnt
conda install -c conda-forge pypy3.6 -y
pypy3 -m ensurepip
pypy3 -m pip install mpmath==1.2.1

# install python packages in environment
pip3 install tensorflow
pip3 install tensorflow-addons tables
conda install -c anaconda pigz==2.4 cffi==1.14.4 -y
conda install -c conda-forge parallel=20191122 zstd=1.4.4 -y
conda install -c conda-forge -c bioconda samtools=1.10 -y
conda install -c conda-forge -c bioconda whatshap=1.4 -y
conda install -c conda-forge xz zlib bzip2 automake curl -y
conda install seaborn
#Go to the installation location of the Attdeepcaller program (download to the specified location and extract the samtools and longphase packages)
Cd Attdeepcaller
#Install libclair3:
make PREFIX=${CONDA_PREFIX}

Train and test the attdeepcaller model:
1.Data preparation
conda activate attdeepcaller
The output union.vcf.gz is placed in the specified folder: OUTPUT_DIR
①sh subsampledata.sh#Downsampled data
②sh rep_uni.sh #Normalized data
2.pileup data training
④sh trainpileupmodel.sh #Training preparation（pileup)
⑤sh pileup_training.sh #Training（pileup)

3.full-alignment training
①sh trainfullalignmodel.sh #Training preparation（full-alignment)
②sh fullalign_training.sh#Training(fullalignment)
4.Testing
①sh clair3_ont_quick_demo.sh #Testing(conda activate attdeepcaller)．
②sh visualization_clair3_ont_quick_demo.sh #Test visualization(conda activate happy-env)

conda activate attdeepcaller

③python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/happy.vcf.gz --output_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/metrics ##Statistical result

5.Model available
The trained model is available at the following link:
Link: https://pan.baidu.com/s/1fO5mlrko5KhdH4lwXVr_Bw
Extraction code: glcv

6. Data available

Reference genomes
GRCh38_no_alt
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz

View in Supplementary Materials.

