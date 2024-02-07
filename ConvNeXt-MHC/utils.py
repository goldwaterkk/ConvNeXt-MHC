import math
import pandas as pd
import numpy as np
import os, gc, re
import tensorflow as tf
from tqdm import tqdm


def data_load_5folds(train_x, train_y):
    path = "./Trans2Input9/Input9_data"
    years_data = []
    # path = "E:\\联想软件\\SCU\\0-CNN-Finall\\1-Data\\npy\\No_smith\\data_double"
    path = "E:\\联想软件\\SCU\\0-CNN-Finall\\1-Data\\npy\\Convnet"
    for i in range(years_1, years_2 + 1):
        try:
            years_data.append(np.load(path + "\\{}.npy".format(i)))
        except:
            continue
    years_data = np.concatenate(tuple(years_data), axis=0)
    np.random.shuffle(years_data)
    x = years_data[:, :, :-1]
    x = x.reshape(-1, 329, 14, 1)
    y = years_data[:, 2, -1]
    # y = tf.keras.utils.to_categorical(y)
    return x, y


def data_load(i):
    path = "./data/npy/ms_train/train_{}.npz".format(i)
    data = np.load(path)
    x_data = data["x"]
    y_data = data["y"]
    temp_y = []
    for i in y_data:
        temp_y.append([1 - i, i])
    return x_data, np.array(temp_y)


def data_load_af(i):
    path = "./data/npy/af_train/af_{}.npz".format(i)
    data = np.load(path)
    x_data = data["x"]
    y_data = data["y"]
    temp_y = []
    for i in y_data:
        temp_y.append([1 - i, i])
    return x_data, np.array(temp_y)


def data_load_ms(i):
    path = "./data/npy/ms_train/train_{}.npz".format(i)
    data = np.load(path)
    x_data = data["x"]
    y_data = data["y"]
    temp_y = []
    for i in y_data:
        if (i >= 0.425):
            temp_y.append(1)
        else:
            temp_y.append(0)
    return x_data, np.array(temp_y)


def data_load_ms_independent():
    path = "./data/npy/ms_valid/ms_valid.npz"
    data = np.load(path)
    x_data = data["x"]
    y_data = data["y"]
    temp_y = []
    for i in y_data:
        if (i >= 0.425):
            temp_y.append(1)
        else:
            temp_y.append(0)
    return x_data, np.array(temp_y)


def generate_MHC_ms_independent(batch_size=32):
    train_x, train_y = data_load_ms_independent()  # X=(-1,21,9,20)  Y=(-1,2) [0.2 , 0.4]
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(buffer_size=train_y.shape[0])
    if (batch_size != 0):
        train_ds = train_ds.batch(batch_size)
    del train_x
    del train_y
    return train_ds


def generate_MHC_15(X, Y, batch_size=32):
    train_x = np.load("./Trans2Input9/Input15_data/{}".format(X))
    train_y = np.load("./Trans2Input9/Input15_data/{}".format(Y))
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(buffer_size=train_y.shape[0])
    if (batch_size != 0):
        train_ds = train_ds.batch(batch_size)
    del train_x
    del train_y
    return train_ds


def generate_MHC_iedb(X, Y):
    train_x = np.load("./Trans2Input9/iedb_npy_9/{}".format(X))
    train_y = np.load("./Trans2Input9/iedb_npy_9/{}".format(Y))
    return train_x, train_y


def generate_MHC_5folds_15(batch_size=32, number=[0, 1, 2, 3]):
    data_x = []
    data_y = []
    for i in number:
        data_x.append(np.load("./Trans2Input9/Input15_data/X_{}.npy".format(i)))
        data_y.append(np.load("./Trans2Input9/Input15_data/Y_{}.npy".format(i)))
    data_x = np.concatenate(tuple(data_x), axis=0)
    data_y = np.concatenate(tuple(data_y), axis=0)
    train_ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    train_ds = train_ds.shuffle(buffer_size=data_y.shape[0])
    if (batch_size != 0):
        train_ds = train_ds.batch(batch_size)
    return train_ds


def generate_MHC_5folds_9(batch_size=32, number=[0, 1, 2, 3]):
    data_x = []
    data_y = []
    for i in number:
        data_x.append(np.load("./Trans2Input9/Input9_data/X_{}_AP.npy".format(i)))
        data_y.append(np.load("./Trans2Input9/Input9_data/Y_{}_AP.npy".format(i)))
    data_x = np.concatenate(tuple(data_x), axis=0)
    data_y = np.concatenate(tuple(data_y), axis=0)
    train_ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    train_ds = train_ds.shuffle(buffer_size=data_y.shape[0])
    if (batch_size != 0):
        train_ds = train_ds.batch(batch_size)
    return train_ds


def cosine_rate(now_step, total_step, end_lr_rate):
    rate = ((1 + math.cos(now_step * math.pi / total_step)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
    return rate


def cosine_scheduler(initial_lr, epochs, steps, warmup_epochs=1, end_lr_rate=1e-7, train_writer=None):
    """custom learning rate scheduler"""
    assert warmup_epochs < epochs
    warmup = np.linspace(start=1e-8, stop=initial_lr, num=warmup_epochs * steps)
    remainder_steps = (epochs - warmup_epochs) * steps
    cosine = initial_lr * np.array([cosine_rate(i, remainder_steps, end_lr_rate) for i in range(remainder_steps)])
    lr_list = np.concatenate([warmup, cosine])

    for i in range(len(lr_list)):
        new_lr = lr_list[i]
        if train_writer is not None:
            # writing lr into tensorboard
            with train_writer.as_default():
                tf.summary.scalar('learning rate', data=new_lr, step=i)
        yield new_lr


def get_predict(Pre_y):
    Y = tf.keras.layers.Softmax()(Pre_y)  # Trans the logist predict value to label
    return np.argmax(Y, axis=1)


def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


def get_accuracy(conf_matrix):
    """
    Calculates accuracy metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    return (TP + TN) / (TP + FP + FN + TN)


def get_precision(conf_matrix):
    """
    Calculates precision metric from the given confusion matrix.
    """
    TP, FP = conf_matrix[0][0], conf_matrix[0][1]

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return 0


def get_recall(conf_matrix):
    """
    Calculates recall metric from the given confusion matrix.
    """
    TP, FN = conf_matrix[0][0], conf_matrix[1][0]

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 0


def get_f1score(conf_matrix):
    """
    Calculates f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def get_mcc(conf_matrix):
    """
    Calculates Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    if TP + FP > 0 and TP + FN > 0 and TN + FP > 0 and TN + FN > 0:
        return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        return 0


def evaluate(Y_pred, Y_real):
    conf_matrix = get_confusion_matrix(Y_real, Y_pred)
    precision = get_precision(conf_matrix)
    recall = get_recall(conf_matrix)
    fscore = get_f1score(conf_matrix)
    mcc = get_mcc(conf_matrix)
    val_acc = get_accuracy(conf_matrix)

    return precision, recall, fscore, mcc, val_acc


def Format_Y(Pre_y, Test_y):
    y1 = []
    y2 = []
    for i in range(Pre_y.shape[0]):
        if (Pre_y[i][0] >= 0.5):
            y1.append(1)
        else:
            y1.append(0)
        if (Test_y[i][0] == 1):
            y2.append(1)
        else:
            y2.append(0)
    return np.array(y1), np.array(y2)


def Format_Y2(Pre_y, Test_y):
    y1 = []
    y2 = []
    for i in range(Pre_y.shape[0]):
        if (Pre_y[i][0] >= 0.5):
            y1.append(0)
        else:
            y1.append(1)
        if (Test_y[i][0] == 1):
            y2.append(0)
        else:
            y2.append(1)
    return np.array(y1), np.array(y2)


def _parse_function(example_proto):
    features = {"data": tf.io.FixedLenFeature((), tf.string),
                "label": tf.io.FixedLenFeature((), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    data = tf.io.decode_raw(parsed_features['data'], tf.float32)
    return data, parsed_features["label"]


def readProp(filename):
    '''
    读取特征向量
    '''
    prop = pd.read_csv(filename, sep=',', encoding='utf-8-sig', header=0, index_col=0)
    # 按列标准化(归一化)
    # prop = prop.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
    # prop = prop.astype(np.float32)
    # print(prop)
    # print(prop.index.to_list())
    return prop


def readSeqInfo(filename):
    '''
    读取伪序列的序列信息
    '''
    fr = open(filename)
    raw_datas = fr.readlines()

    seq_dict = {}

    for i in range(len(raw_datas)):

        if raw_datas[i].startswith('>'):
            label = raw_datas[i].split(' ')[0].replace('>HLA_', '')  # 第一行以空格分隔 第0列是HLA名字将>HLA_去掉，len(raw_datas)是文件总行数
            seq = raw_datas[i + 1].replace('\n', '')  # 第二行伪序列

            seq_dict[label] = seq

    return seq_dict


def readSiteInfo(filename):
    '''
    读取伪序列的残基编号
    '''
    fr = open(filename)
    raw_datas = fr.readlines()

    site_dict = {}

    for line in raw_datas:
        lineList = line.split('=')
        label = lineList[0].replace(' ', '')
        value = lineList[1].replace(' ', '').replace('\n', '')
        value = eval(value)

        site_dict[label] = value

    return site_dict


def trans_AllData_3D(dataset, prop_pep, prop_pseudo, siteInfo, seq_dict, HLA):
    '''
    dataset为遍历单序列的集合，
    读取MHCz_HLA_affinity.csv文件中的序列信息并完成编码
    pep_len定义为15
    '''
    # 9肽中的9个残基位点分别对应的接触残基编号liuyang
    HLA_pseudoSeq = {0: [7, 59, 62, 63, 159, 163, 167, 171],
                     1: [7, 9, 24, 45, 63, 65, 66, 67, 99, 159],
                     2: [66, 99, 114, 156, 159],
                     3: [62, 65, 66, 69, 163],
                     4: [69, 70, 114, 155, 156],
                     5: [70, 73, 114, 156],
                     6: [73, 74, 77, 147, 150, 152, 155, 156],
                     7: [73, 76, 77, 80, 143, 146, 147],
                     8: [77, 80, 81, 84, 95, 97, 116, 123, 143, 146, 147],
                     }

    # dataset = pd.read_csv(data_file)
    HLA_pseudoSeq_set = set()  # 结果去重
    for value in HLA_pseudoSeq.values():  # value[77,80,81,84...]
        for i in value:
            HLA_pseudoSeq_set.add(i)

    # 接触位点  7、9、24...
    HLA_pseudoSeq_list = list(HLA_pseudoSeq_set)
    HLA_pseudoSeq_list.sort()
    # 等位基因
    seq_dict_keys = list(seq_dict.keys())
    pep_len = 9
    # peptides = []
    # labels = []
    labels = []
    encode_matix = []
    aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aa_pos_dict = dict()  # 创建一个空字典  len(dataset)
    f = open("./not_9mer_error_data.txt", "w")
    # print(dataset[1])
    for i in tqdm(range(len(dataset))):

        peptide = dataset[i]

        # allel 标记的是 MHC
        allele = re.sub('[*:]', '', HLA[i])
        pep_len_ = len(peptide)  # peptide长度
        if (pep_len_ != 9):
            f.write("{} : {}".format(i, peptide))
            continue
        # print(peptide)
        if (allele in seq_dict_keys):
            one_matrix = []

            # 获取相应等位基因的残基-位点对应关系: aa_pos_dict字典{aa：position}
            # 残基列表： aa_list = A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
            for aa in aa_list:
                aa_pos_dict[aa] = []
            # print(aa_pos_dict)
            for resi in HLA_pseudoSeq_list:  # resi在[7,9,24,45...]中
                aa_pos_dict[seq_dict[allele][siteInfo.index(resi)]].append(
                    resi)  # index 返回指定值首次出现的位置 seq_dict:HLA_pseudoSeqs.fasta siteinfo: site.txt中HLA_pseudo_sequence[7,9,24,...]
            # print(aa_pos_dict)  MHC氨基酸对应的残基位置
            for aa in aa_list:
                matrix = []
                for pos in range(pep_len):  # pep_len=9
                    row = []
                    # 此处默认所有为转化后的多肽，空位用“-”表示，由于blank进行了判断，此处的“-”并未有实际作用，仅仅为了数据正确，若没有blank，则置blank为-1
                    if peptide[pos] == 'X' or peptide[pos] == '-':
                        row = [0 for i in range(prop_pep.shape[1] + prop_pseudo.shape[1])]
                    else:
                        if list(set(aa_pos_dict[aa]).intersection(set(HLA_pseudoSeq[pos]))):
                            row_pep = prop_pep.loc[peptide[pos]].values
                            row_pseudo = prop_pseudo.loc[aa].values
                            row = np.append(row_pseudo, row_pep)
                        else:
                            row = [0 for i in range(prop_pep.shape[1] + prop_pseudo.shape[1])]
                    matrix.append(row)
                one_matrix.append(matrix)
            encode_matix.append(one_matrix)
        else:
            print('cant predict {}'.format(allele))
    f.close()
    encode_matix = np.array(encode_matix)
    del dataset, peptide, HLA
    gc.collect()
    return encode_matix


def input_matrix_generation(csv_file_name="af_valid_data.csv"):
    print("start to generate data path :", "./data_set/", csv_file_name)
    prop_pep = readProp('./Trans2Input9/onehot_new.csv')
    prop_pseudo = readProp('./Trans2Input9/pseudo_encoding.csv')
    site_dict = readSiteInfo("./Trans2Input9/pseudo/site.txt")
    seq_dict = readSeqInfo("./Trans2Input9/pseudo/HLA_pseudoSeqs.fasta")
    pseudo_length = len(site_dict['HLA_pseudo_sequence'])
    # temp_ms = pd.read_csv("../data_set/af_valid_data.csv")
    temp_file = pd.read_csv("./data_set/" + csv_file_name)  # ("../data_set/af_valid_data.csv")
    csv_pep = temp_file['9mer']
    csv_allele = temp_file['allele']
    X_t = trans_AllData_3D(csv_pep, prop_pep, prop_pseudo, site_dict['HLA_pseudo_sequence'], seq_dict,
                           csv_allele)
    return X_t


def divid_to_csv():
    csv_dataset = pd.read_csv("./data/csv/ms_train_shuffle.csv")
    per_length = 80000
    times = 0
    for i in range(per_length, len(csv_dataset), per_length):  # 拆分为csv文件
        csv_dataset[i - per_length:i].to_csv("./data/csv/ms_train/ms_{}.csv".format(times), index=False)
        times = times + 1
    else:
        csv_dataset[i:].to_csv("./data/csv/ms_train/ms_{}.csv".format(times), index=False)
        times = times + 1
