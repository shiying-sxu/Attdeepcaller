环境安装最后放到了attdeepclair下，程序代码在clair3-main202302

训练并测试attdeepcaller模型：
（一）数据准备
在原始环境attdeepcaller下运行
conda activate attdeepcaller
输出的unified.vcf.gz放到了指定的文件夹：OUTPUT_DIR="/work/Clair3-main-sy/data/outputref-HG004_GRCh38"
①sh subsampledata.sh下采样数据
②sh rep_uni.sh 归一化数据
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 11行&59行)，要对应到具体文件夹
③sh rep_uni.sh 下采样数据归一化(归一化比例为600，也可为其他，对应修改)
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 60行)，要对应到具体文件夹
（二）pileup data 训练
在原始环境 attdeepcaller下运行
④sh trainpileupmodel.sh 训练准备（pileup)
在环境 attdeepcaller下运行
⑤sh pileup_training.sh 训练（pileup)
训练pileup模型
（三）full-alignment 训练
sh fullaligndatapre.sh 数据准备(full-alignment)
①sh trainfullalignmodel.sh 训练准备（full-alignment)
②sh fullalign_training.sh 训练fullalignment模型
（四）测试
①将 pileup模型和full-alignment模型放到一个指定文件夹为测试作准备,clair3_ont_quick_demo.sh文件47行 model_path=/work/Clair3-main-sy/data/mytrainmodel/ont
(训练好的模型路径:/work/Clair3-main-sy/data/HG001-600-1000-rainoutput/pileup/modeltest/train-BILSTM-CBAMRESNET/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为pileup.data-00000-of-00001 和pileup.index
full_alignment同理 /work/Clair3-main-sy/data/HG001-600-1000-trainoutput/fullalignment/modeltest/train-BILSTM-CBAMRESNEXT/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为full_alignment.data-00000-of-00001 和full_alignment.index)
②sh clair3_ont_quick_demo.sh 测试(环境依然是attdeepcaller)．
③sh visualization_clair3_ont_quick_demo.sh测试可视化( 环境切换到conda activate happy-env)
④在2visualization_clair3_ont_quick_demo.sh中运行一行代码，实现最终测试的结果，放在指定文件夹的metrics文件里
conda activate attdeepcaller
运行命令：
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/happy.vcf.gz --output_fn=/work/Clair3-main-sy/clair3_ont_quickDemo/output/metrics

#HG001
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output/metrics
#HG001original
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-HG001/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-HG001/metrics

#HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-new2-HG002/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-new2-HG002/metrics
#600HG002-original
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-600HG002/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-600HG002/metrics
#600HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-new1-600HG002/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-new1-600HG002/metrics
##HG003
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-HG003-new/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-HG003-new/metrics
##HG003-original
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-HG003-original/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-HG003-original/metrics
#HG004
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-HG004/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/test/output-original-HG004/metrics



HIFI模型：
训练并测试attdeepcaller模型：
（一）数据准备
在原始环境attdeepcaller下运行
conda activate attdeepcaller
输出的unified.vcf.gz放到了指定的文件夹：OUTPUT_DIR="/work/Clair3-main-sy/data/datatest/Pacbiohifi/outputref-HG002_GRCh38"
①sh subsampledata.sh下采样数据
②sh rep_uni_hifi.sh 归一化数据
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 11行&59行)，要对应到具体文件夹
③sh rep_uni_hifi.sh 下采样数据归一化(归一化比例为600，也可为其他，对应修改)
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 60行)，要对应到具体文件夹
（二）pileup data 训练
在原始环境 attdeepcaller下运行
④sh trainpileupmodel_hifi.sh 训练准备（pileup hifi)
在环境 attdeepcaller下运行
⑤sh pileup_hifi_training.sh 训练（pileup hifi)
训练pileup模型
（三）full-alignment 训练
sh trainfullalignment_hifi_r1.sh
sh 2trainfullalignment_hifi_r1.sh


sh fullaligndatapre_hifi.sh 数据准备(full-alignment)
①sh trainfullalignmodel_hifi.sh 训练准备（full-alignment)
②sh fullalign_hifi_training.sh 训练fullalignment模型
（四）测试
illumina需要在环境中安装realigner软件包
# Install boost library
conda install -c conda-forge boost=1.67.0 -y
#echo "Environment:" ${CONDA_PREFIX}
echo "Environment:" /home/user/anaconda3/envs/attdeepcaller
# Make sure in Clair3 directory
cd Clair3

cd preprocess/realign
g++ -std=c++14 -O1 -shared -fPIC -o realigner ssw_cpp.cpp ssw.c realigner.cpp
g++ -std=c++11 -shared -fPIC -o debruijn_graph -O3 debruijn_graph.cpp -I /home/user/anaconda3/envs/attdeepcaller/include -L /home/user/anaconda3/envs/attdeepcaller/lib
①将HIFI数据训练的 pileup模型和full-alignment模型放到一个指定文件夹为测试作准备,clair3_ont_quick_demo.sh文件47行 model_path=/work/Clair3-main-sy/data/mytrainmodel/hifi
(训练好的模型路径:/work/Clair3-main-sy/data/HG001-HG002-trainoutput/hifi/pileup/modeltest/train-pileup-resnext-CBAM-bilstm0.0333/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为pileup.data-00000-of-00001 和pileup.index
full_alignment同理 /work/Clair3-main-sy/data/HG001-HG002-trainoutput/hifi/fullalign/modeltest/train-fullalign-resnext-CBAM-bilstm/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为full_alignment.data-00000-of-00001 和full_alignment.index)
②sh clair3_hifi_quick_demo.sh 测试(环境依然是attdeepcaller)．
tabix -fp vcf HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
报错Command 'vcfcheck output.vcf.gz --check-bcf-errors 1' returned non-zero exit status 1
发现merge_out.vcf多了一行,删除就行(2594行)
vcf文件压缩并建立索引命令:
cd /work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG002/
bgzip merge_output.vcf
tabix -fp vcf merge_output.vcf.gz

samtools index /work/Clair3-main/data/datatest/Pacbiohifi/HG001/HG001_GRCh38_HIFI.bam
③sh visualization_clair3_hifi_quick_demo.sh测试可视化( 环境切换到conda activate happy-env)
④运行以下代码，实现最终测试的结果，放在指定文件夹的metrics文件里
conda activate attdeepcaller
运行命令：
#chr20
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main/data/datatest/clair3_pacbio_hifi_quickDemo/output/happy.vcf.gz --output_fn=/work/Clair3-main/data/datatest/clair3_pacbio_hifi_quickDemo/output/metrics
#hg001
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG001-1217/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG001-1217/metrics
#HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG002-1218/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG002-1218/metrics
#HG003
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG003-1218/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG003-1218/metrics
#HG004
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG0041215/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG0041215/metrics
#HG005
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG005-1215/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-HG005-1215/metrics
#33xHG002-GRCh37
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-33xHG002-1210/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/hifitest/output-33xHG002-1210/metrics


Illumina模型：
download data

训练并测试attdeepcaller模型：
（一）数据准备
在原始环境attdeepcaller下运行
conda activate attdeepcaller
输出的unified.vcf.gz放到了指定的文件夹：OUTPUT_DIR="/work/Clair3-main-sy/data/datatest/Illumina/outputref-HG001_GRCh38"
①sh subsampledata.sh下采样数据
②sh rep_uni_illumina.sh 归一化数据
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 11行&59行)，要对应到具体文件夹
③sh rep_uni_illumina.sh 下采样数据归一化(归一化比例为600，也可为其他，对应修改)
在这里生成了unified.vcf.gz文件，在训练中会用到(trainmymodel 60行)，要对应到具体文件夹
（二）pileup data 训练
在原始环境 attdeepcaller下运行
④sh trainpileupmodel_illumina.sh 训练准备（pileup illumina)
在环境 attdeepcaller下运行
⑤sh pileup_illumina_training.sh 训练（pileup illumina)
训练pileup模型
（三）full-alignment 训练
sh trainfullalignment_illumina_r1.sh
sh 2trainfullalignment_illumina_r1.sh

（四）测试
①将illumina数据训练的 pileup模型和full-alignment模型放到一个指定文件夹为测试作准备,clair3_ont_quick_demo.sh文件47行 model_path=/work/Clair3-main-sy/data/mytrainmodel/illumina
(训练好的模型路径:/work/Clair3-main-sy/data/HG001-HG002-trainoutput/illumina/pileup/modeltest/train-pileup-resnext-CBAM-bilstm0.0333/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为pileup.data-00000-of-00001 和pileup.index
full_alignment同理 /work/Clair3-main-sy/data/HG001-HG002-trainoutput/hifi/fullalign/modeltest/train-fullalign-resnext-CBAM-bilstm/best_val_loss/variables/,将variables.data-00000-of-00001 和
variables.index文件复制到指定文件夹，并把名字改为full_alignment.data-00000-of-00001 和full_alignment.index)
②sh clair3_illumina_quick_demo.sh 测试(环境依然是attdeepcaller)．sh clair3_ilmn_quick_demo.sh
③sh visualization_clair3_illumina_quick_demo.sh测试可视化( 环境切换到conda activate happy-env)
④运行以下代码，实现最终测试的结果，放在指定文件夹的metrics文件里
conda activate attdeepcaller
运行命令：
#chr20
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main/data/datatest/clair3_pacbio_illumina_quickDemo/output0110/happy.vcf.gz --output_fn=/work/Clair3-main/data/datatest/clair3_pacbio_illumina_quickDemo/output0110/metrics
#hg001
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG001/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG001/metrics
#HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG002/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG002/metrics
#HG003
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG003/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG003/metrics
#HG004
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG004/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG004/metrics
#HG005
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG005/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG005/metrics
#52X HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-52X-HG002-0217/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-52X-HG002-0217/metrics


#0110trainning on HG002
python GetOverallMetrics.py --happy_vcf_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG002-0110/happy.vcf.gz --output_fn=/work/Clair3-main-sy/data/mytrainmodel/illuminatest/output-HG002-0110/metrics
ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_SequelII_CCS_11kb/HG002.SequelII.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
axel -a -n 40 https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V2.3.4_2019-06-26/ultra-long-ont_GRCh38_reheader.bam
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/ChineseTrio/HG005_NA24631_son/PacBio_SequelII_CCS_11kb/HG005.SequelII.pbmm2.hs37d5.whatshap.haplotag.10x.bam.bai
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh37/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh37/HG005_GRCh37_1_22_v4.2.1_benchmark.bed
https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh37/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz.tbi