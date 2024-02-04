import pandas as pd
import os
import argparse
def one2many(seq):
    dataset = []
    blank = []
    if (len(seq) == 8):
        for i in range(9):
            ans = []  # 有空位的九肽
            for k in range(8):
                if (k == i):
                    ans.append('-')
                ans.append(seq[k])
            if (i == 8):
                ans.append('-')
            dataset.append(''.join(ans))  # 转化为字符串，添加进集合
        blank = [0, 1, 2, 3, 7, 8, 12, 13, 14]  # 由于原本模型转化为15肽，进行了位置偏移矫正
        return dataset, blank

    elif (len(seq) == 9):
        dataset = [seq]
        blank = [-1]
        return dataset, blank
    else:
        # 大于9 ， 随机摘除位点,暂时不考虑
        for i in range(len(seq) - 9 + 1):  # 例如10肽，生成两条9肽，因此 len - 9 + 1 , i作为基石 ，两边摘除
            ans = []
            for k in range(9):
                ans.append(seq[i + k])
            dataset.append(''.join(ans))
            blank.append(-1)
        for i in range(1, 9):
            ans = []
            diff = len(seq) - 9 - 1
            for k in range(len(seq)):
                if (k > i + diff):
                    ans.append(seq[k])
                elif (k < i):
                    ans.append(seq[k])
            dataset.append(''.join(ans))
            blank.append(-1)
    return dataset, blank

def save_to_pep(temp,file_name):
    f = open("./tmp/{}.pep".format(file_name),"w")
    for i in temp:
        f.write(i+"\n")
    f.close()
    
def select_max_value(file_name):
    pep = "test"
    with open('./tmp/{}.xls'.format(file_name), 'r') as file:
        lines = file.readlines()
    
    with open('./tmp/{}.tsv'.format(file_name), 'w') as temp_file:
        temp_file.writelines(lines[1:])
    
    # 使用 Pandas 读取临时文件
    df = pd.read_csv('./tmp/{}.tsv'.format(file_name), sep='\t')
    max_score = df["BA-score"].max()
    
    row = df[df["BA-score"] == max_score]
    row = row.iloc[0]
    return row['Peptide']
    
def main():
    parser = argparse.ArgumentParser(description="Your file name test.csv, need 'peptide', 'allele', '9mer'")
    parser.add_argument('file_name', help="Name of the CSV file")
    args = parser.parse_args()
    file_name = args.file_name   # You file name test.csv , need peptide,allele,9mer
    df = pd.read_csv("./"+file_name)
    df['9mer'] = ""
    for index, row in df.iterrows():
        pep = row['peptide']
        alle = row['allele']
        if(len(pep) == 9):
            df.at[index, "9mer"] = pep
        else:
            peps , _ =one2many(pep)
            save_to_pep(peps,file_name)
            os.system("./netMHCpan -p ./tmp/{}.pep -BA -xls -a {} -xlsfile ./tmp/{}.xls".format(file_name,alle[0:7]+":"+alle[7:],file_name))
            print("./netMHCpan -p ./tmp/{}.pep -BA -xls -a {} -xlsfile ./tmp/{}.xls".format(file_name,alle[0:7]+":"+alle[7:],file_name))
            temp = select_max_value(file_name)
            df.at[index, "9mer"] = temp
    df.to_csv("output.csv", index=False)
    
main()
# python Gen9mer.py your_file.csv