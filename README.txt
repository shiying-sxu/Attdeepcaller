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
②sh 0_rep_uni.sh #Normalized data
2.pileup data training
④sh 1_trainpileupmodel.sh #Training preparation（pileup)
⑤sh 2_pileup_training.sh #Training（pileup)


3.full-alignment training
①sh 3_trainfullalignment_ont_r1.sh #Training preparation（full-alignment)
②sh 4_training_fullalignment_ont_r1.sh#Training(fullalignment)

4.Testing
①sh 5_attdeepcaller_ont_quick_demo-HG002.sh #Testing(conda activate attdeepcaller)．
conda activate happy-env
②sh 6_visualization_attdeepcaller_ont_quick_demo-HG002.sh #Test visualization(conda activate happy-env)

conda activate attdeepcaller

③python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/happy.vcf.gz --output_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/metrics ##Statistical result

5.Model available
The trained model is available at the following link:
Link: https://pan.baidu.com/s/1pIcnAGP17T9fFXIBiSGB3A?pwd=wvyu

Extraction code: wvyu

6. Data available

Reference genomes
GRCh38_no_alt
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz

7.Illumina data preprocessing
illumina need to install realigner Installation package
# Install boost library
conda install -c conda-forge boost=1.67.0 -y
#echo "Environment:" ${CONDA_PREFIX}
echo "Environment:" /home/user/anaconda3/envs/attdeepcaller
# Make sure in Attdeepcaller directory
cd Attdeepcaller

cd preprocess/realign
g++ -std=c++14 -O1 -shared -fPIC -o realigner ssw_cpp.cpp ssw.c realigner.cpp
g++ -std=c++11 -shared -fPIC -o debruijn_graph -O3 debruijn_graph.cpp -I /home/user/anaconda3/envs/attdeepcaller/include -L /home/user/anaconda3/envs/attdeepcaller/lib





View in Supplementary Materials.

