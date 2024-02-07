import numpy as np
import pandas as pd
import tensorflow as tf
from mhcflurry import amino_acid
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.encodable_sequences import EncodableSequences
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from utils import *


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


def readSeqInfo2(filename):
    '''
    读取伪序列的序列信息 MHCflurry, NetMHCpan4.0
    '''
    fr = open(filename)
    raw_datas = fr.readlines()

    seq_dict = {}

    for i in range(len(raw_datas)):
        temp = raw_datas[i].split(' ')
        seq_dict[temp[0]] = temp[1].strip()

    return seq_dict


'''
    encoding = AlleleEncoding(
        ["A*02:01"],
        {
            "A*02:01": "ACDD",
        }
    )
'''


def Concat_reshape(data1, data2):
    data3 = np.concatenate((data1, data2), axis=1)
    data3 = np.expand_dims(data3, axis=3)
    data3 = np.concatenate((data3, data3, data3, data3), axis=3)
    return data3


def MHCflurry_input_matrx(filename, file_data):  # (36+15)*21*4
    seq_dict = readSeqInfo2(filename)
    df = pd.read_csv(file_data)
    encoding = AlleleEncoding(
        alleles=df['allele'].to_list(),
        allele_to_sequence=seq_dict,
        borrow_from=None
    )
    encoder_alle = encoding.fixed_length_vector_encoded_sequences("BLOSUM62")

    encoded_peptides = EncodableSequences.create(df['peptide'].to_list())
    encoder_peptides = encoded_peptides.variable_length_to_fixed_length_vector_encoding("BLOSUM62")

    X = Concat_reshape(encoder_peptides, encoder_alle)
    Y = np.array(df['NB'].to_list())
    return X, Y


def Degenerate_code(i):  # 20*9*21
    tt = np.load("./data/npy/ms_train/train_{}.npz".format(i))
    return tt['x'], tt['y']


def NetMHCpan4(filename, file_data):
    df = pd.read_csv(file_data)
    seq_dict = readSeqInfo2(filename)
    index_encoded_matrix = amino_acid.index_encoding(df['9mer'].to_list(), amino_acid.AMINO_ACID_INDEX)
    encoder_peptide = amino_acid.fixed_vectors_encoding(index_encoded_matrix,
                                                        amino_acid.ENCODING_DATA_FRAMES["BLOSUM62"])

    alle = []
    temp = df['allele'].to_list()
    for i in range(len(temp)):
        alle.append(seq_dict[temp[i]])

    index_encoded_matrix = amino_acid.index_encoding(alle, amino_acid.AMINO_ACID_INDEX)
    encoder_alle = amino_acid.fixed_vectors_encoding(index_encoded_matrix, amino_acid.ENCODING_DATA_FRAMES["BLOSUM62"])
    # (90803, 34, 21)
    X = Concat_reshape(encoder_peptide, encoder_alle)
    Y = np.array(df['NB'].to_list())
    return X, Y


def model1_mhcflurry(num_classes=2):
    input_shape1 = (49, 21, 4)
    input1 = Input(shape=input_shape1)
    conv1_1 = Conv2D(16, kernel_size=(2, 2), activation='relu')(input1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool1_1)
    pool1_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1_2)
    flat1 = Flatten()(pool1_2)
    dense1 = Dense(64, activation='softmax')(flat1)
    output_layer = Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=[input1], outputs=output_layer)
    model.compile(loss="binary_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def model2_NetMHCpan4(num_classes=2):
    input_shape2 = (43, 21, 4)

    input2 = Input(shape=input_shape2)
    conv2_1 = Conv2D(16, kernel_size=(2, 2), activation='relu')(input2)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2_1)
    conv2_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool2_1)
    pool2_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2_2)
    flat2 = Flatten()(pool2_2)
    dense1 = Dense(64)(flat2)
    output_layer = Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=[input2], outputs=output_layer)
    model.compile(loss="binary_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def model3_DE_CNN(num_classes=2):
    input_shape3 = (20, 9, 21)

    input3 = Input(shape=input_shape3)
    conv3_1 = Conv2D(16, kernel_size=(2, 2), activation='relu')(input3)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3_1)
    conv3_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool3_1)
    pool3_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3_2)
    flat3 = Flatten()(pool3_2)
    dense1 = Dense(64)(flat3)
    output_layer = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=[input3], outputs=output_layer)
    model.compile(loss="binary_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def DE_CNN_AF(num_classes=2):
    input_shape3 = (20, 9, 21)

    input3 = Input(shape=input_shape3)
    conv3_1 = Conv2D(16, kernel_size=(2, 2), activation='relu')(input3)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3_1)
    conv3_2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool3_1)
    pool3_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3_2)
    flat3 = Flatten()(pool3_2)
    dense1 = Dense(64)(flat3)
    output_layer = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=[input3], outputs=output_layer)
    model.compile(loss="mse",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def train(index, times):
    file_path_alle_NetMHCpan = "./Trans2Input9/pseudo/NetMHCpan_pseudo"
    file_path_alle_flurry = "./Trans2Input9/pseudo/MHCflurry_pseudosequences"

    model_flurry = model1_mhcflurry(2)
    model_Net = model2_NetMHCpan4(2)
    model_conext = model3_DE_CNN(2)

    for k in range(1):
        for i in index:
            print("---------{}--------".format(i))
            file_data = "./data/csv/ms_train/ms_{}.csv".format(i)

            X, Y = MHCflurry_input_matrx(file_path_alle_flurry, file_data)
            Y = tf.one_hot(Y, 2)
            model_flurry.fit(X, Y)

            X, Y = NetMHCpan4(file_path_alle_NetMHCpan, file_data)
            Y = tf.one_hot(Y, 2)
            model_Net.fit(X, Y)

            X, Y = Degenerate_code(i)
            Y = tf.one_hot(Y, 2)
            model_conext.fit(X, Y)

    model_flurry.save_weights(
        "./save_model_weight/input_matrix_compare_5folds_model/model1_mhcflurry_{}.h5".format(times))
    model_Net.save_weights(
        "./save_model_weight/input_matrix_compare_5folds_model/model2_NetMHCpan4_{}.h5".format(times))
    model_conext.save_weights("./save_model_weight/input_matrix_compare_5folds_model/model3_DE_CNN_{}.h5".format(times))


def valid(index, times):
    f = open("./log_input_matrix_compare_5folds.csv", "a")

    file_path_alle_NetMHCpan = "./Trans2Input9/pseudo/NetMHCpan_pseudo"
    file_path_alle_flurry = "./Trans2Input9/pseudo/MHCflurry_pseudosequences"

    model_flurry = model1_mhcflurry(2)
    model_Net = model2_NetMHCpan4(2)
    model_cnn = model3_DE_CNN(2)

    model_flurry.load_weights(
        "./save_model_weight/input_matrix_compare_5folds_model/model1_mhcflurry_{}.h5".format(times))
    model_Net.load_weights(
        "./save_model_weight/input_matrix_compare_5folds_model/model2_NetMHCpan4_{}.h5".format(times))
    model_cnn.load_weights("./save_model_weight/input_matrix_compare_5folds_model/model3_DE_CNN_{}.h5".format(times))

    for i in index:
        print("---------{}--------".format(i))
        file_data = "./data/csv/ms_train/ms_{}.csv".format(i)

        X, Y = MHCflurry_input_matrx(file_path_alle_flurry, file_data)
        temp_Y = model_flurry.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("MHCflurry_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = NetMHCpan4(file_path_alle_NetMHCpan, file_data)
        temp_Y = model_Net.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("NetMHCpan_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = Degenerate_code(i)
        temp_Y = model_conext.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("DE_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))
        f.write("\n")
    f.close()
