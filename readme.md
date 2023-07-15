# Portable Perceptron Network-based Fast Mode Decision for Video-based Point Cloud Compression
This is the official repository of source codes and deployment methods for the paper "Portable Perceptron Network-based Fast Mode Decision for Video-based Point Cloud Compression". In order to reduce its meaning and express it uniformly, the following takes "PPNforV-PCC" as the root directory, for example, the location of "mpeg-pcc-tmc2-netTest" is "/mpeg-pcc-tmc2-netTest"

<b>If you have contacted TMC2 and HM, you can skip subsequent lengthy instructions and directly use the source under “/mpeg-pcc-tmc2-netTest/dependencies/HM-16.20+SCM-8.8/source” to change the source of original dependencies of TMC2 and check the methods described in our paper.  If you are not familiar with the package structure of TMC2, it is strongly recommended that you configure it as described below.</b>

## <b>Resource Link
The program versions used in the experiment are as follows (You can get their official versions through the link after quotation marks): 

1. TMC2-v18.0: https://github.com/MPEGGroup/mpeg-pcc-tmc2/tree/release-v18.0
2. HDRTools-v0.18: https://gitlab.com/standards/HDRTools/-/tree/0.18-dev
3. MPEG test sequence: https://mpeg-pcc.org/index.php/pcc-content-database/

## <b>Content Introduction
Please note that this source code runs on Linux, if runs on Windows is necessary we strongly commend to use visual studio 2019 or later for compilation and running executable files. A brief introduction to the content provided is listed below:  

- __batchProcessing: Store batch files of .sh, which you can run after configuring the input point cloud to check the experimental data in the paper or configure your own caller using this as a reference.

- __output: Store the intermediate files generated by TMC2, where you can check the occupancy map, geometry map and attribute map. It is necessary to keep the intermediate files. If you want to save space by not saving the intermediate files, you need to set "--keepIntermediateFiles=0" in the configuration file (.sh file in "/__batchProcessing").

- __statisticData: The console output is stored in here.

- cfg, bin, external, pccTestSequence: Separately store configuration files, executables, external dependency files, and input test sequences for batch file calls. Note that if the path of the downloaded standard test sequence changes, you may need to adjust the configuration file corresponding to the sequence name under /cfg/sequence. If you use non-standard test sequences, change the configuration file format based on your requirements.

- PPN_training: Store the content related to neural network training, including datasets, trained models and history and python files used for training. We provide two versions: py and ipy.

- mpeg-pcc-tmc2-netTest: PPN for V-PCC program source file. You can find the part we changed by searching "MesksCode" or "MesksTest".

## <b>Input File
You can download the official test sequences provided by MPEG in the 3 resources mentioned above, but for some reasons, they no longer provide the complete sequence of loot, queen, redandblack, solder, and longdress described in the paper, but basketball_player is still provided. If needed you can still download the sequences mentioned in the paper here: http://plenodb.jpeg.org/pc/8ilabs/

Please decompress the obtained test sequence to the "/pccTestSequence" path. For example, for the input point cloud file of the first frame of the soldier sequence, it should be found by "/pccTestSequence/soldier/Ply/soldier_vox10_0536.ply", and the corresponding The normal file should be able to be found via "pccTestSequence/soldier/soldier_n/soldier_vox10_0536_n.ply". If you are using the normal calculated by MeshLab or some others program based on the input file, although it can be run, the compressed PSNR may be different from the PSNR recorded in the official test file, and may not be consistent with our experimental results.