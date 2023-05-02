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
#进入Attdeepcaller的程序安装位置（下载到指定位置，程序中包含samtools和longphase软件包并解压）
Cd Attdeepcaller
#安装libclair3:
make PREFIX=${CONDA_PREFIX}

训练并测试attdeepcaller模型：
（一）数据准备
在原始环境attdeepcaller下运行
conda activate attdeepcaller
输出的unified.vcf.gz放到了指定的文件夹：OUTPUT_DIR
①sh subsampledata.sh下采样数据
②sh rep_uni.sh 归一化数据
（二）pileup data 训练
④sh trainpileupmodel.sh 训练准备（pileup)
⑤sh pileup_training.sh 训练（pileup)
训练pileup模型
（三）full-alignment 训练
①sh trainfullalignmodel.sh 训练准备（full-alignment)
②sh fullalign_training.sh 训练fullalignment模型
（四）测试
①sh clair3_ont_quick_demo.sh 测试(环境依然是attdeepcaller)．
②sh visualization_clair3_ont_quick_demo.sh测试可视化( 环境切换到conda activate happy-env)

conda activate attdeepcaller

③python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/happy.vcf.gz --output_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/metrics
