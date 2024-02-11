# ConvNeXt-MHC
## Abstract

Peptide binding to major histocompatibility complex (MHC) proteins plays a critical role in T-cell recognition and the specificity of the immune response. As a result, accurate prediction of binding peptides is highly important, particularly in the context of cancer immunotherapy applications. There is a significant need to continually improve the existing prediction methods to meet the demands of this field. We developed ConvNeXt-MHC, an MHC class I predictive model that introduces a degenerate encoding approach to enhance well-established panspecific methods and integrates transfer learning and semi-supervised learning methods into the cutting-edge deep learning framework ConvNeXt. Comprehensive benchmark results demonstrate that ConvNeXt-MHC outperforms state-of-the-art methods in terms of accuracy. We expect that ConvNeXt-MHC will help us foster new discoveries in the field of immunoinformatics in the distant future.

## Availability:

 We constructed a user-friendly website at http://www.combio-lezhang.online/predict/, where users can access our data and application. The source code and data for our proposed method can be found at https://github.com/goldwaterkk/ConvNeXt-MHC.git.

## Get the ConvNeXt-MHC  Source

```
git clone https://github.com/goldwaterkk/ConvNeXt-MHC.git
```

The repository is about 200MB

## Required Dependencies

- [python](https://www.python.org/) (3.9.13)
- [numpy](https://numpy.org/) (1.20.3)
- [pandas](https://pandas.pydata.org/) (1.4.4)
- tensorflow(2.8.0)

## Optional Dependencies

CUDA 12.0 (Required for GPU usage)

## Usage

1. download ”netMHCpan-4.1b.Linux.tar.gz“ from https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

2. Install the software according to netMHCpan-4.1. readme in the netMHCpan-4.1b. Linux. tar. gz folder, and test if the software installation was successful

```
cat netMHCpan-4.1.<unix>.tar.gz | uncompress | tar xvf -
```



3. Modify the value of NMHOME in netMHCpan to change the specific content to the path content of the current 'pwd'

```
vi netMHCpan-4.1/netMHCpan

# full path to the NetMHCpan 4.0 directory (mandatory)
setenv	NMHOME  YOUT_PATH_FROM_PWD
```



4. Place the file Gen9mer.py in the same level directory as the software netMHCpan in the directory of netMHCpan-4.1

```
.
├── data
├── Gen9mer.py
├── Linux_x86_64
├── netMHCpan
├── netMHCpan.1
├── netMHCpan-4.1.readme
├── output.csv
├── test
├── test_data.csv
└── tmp
# need directory of tmp
```



5. After installing argparse, place the files that need to be converted to 9mer in the same level directory and use the command "Python Gen9mer. py test. csv" to obtain output.csv, where the 9mer column meets the core peptide requirements

```
pip install argparse				
python Gen9mer.py test_data.csv		 # 'peptide','allele',''
```

6. Using run_model.py predicts specific BA and AP values
