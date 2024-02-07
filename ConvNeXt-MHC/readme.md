1. download  ”netMHCpan-4.1b.Linux.tar.gz“ from   https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

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