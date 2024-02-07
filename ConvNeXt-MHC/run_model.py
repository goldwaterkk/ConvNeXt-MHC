from utils import input_matrix_generation, get_predict
from model import *
from utils import *


def run_model():
    # 数据保存在./data_set中/，为了节约时间，已经将多肽转换为了9mer数据
    # 如果需要添加新的，参照example文件进行添加，并修改下述文件名
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # BA sore predict

    data_file_name = "af_valid_data.csv"
    input_matrix_BA = input_matrix_generation(data_file_name)
    ConvNeXt_MHC_BA = convnext_10_10(2)
    ConvNeXt_MHC_BA.build((1, 20, 9, 21))
    ConvNeXt_MHC_BA.load_weights("./save_model_weight/ConvNeXt-MHC_BA.h5")
    BA_ans = ConvNeXt_MHC_BA.predict(input_matrix_BA)[:, 1]
    print(BA_ans)

    # AP sore predict
    data_file_name ="ms_valid_data.csv"
    input_matrix_AP = input_matrix_generation(data_file_name)
    ConvNeXt_MHC_AP = convnext_10_10(2)
    ConvNeXt_MHC_AP.build((1, 20, 9, 21))
    ConvNeXt_MHC_AP.load_weights("./save_model_weight/ConvNeXt-MHC_AP.h5")
    AP_ans = get_predict(ConvNeXt_MHC_AP.predict(input_matrix_AP))
    f = open("./test.txt","w")
    for AP in AP_ans:
        f.write("{}\n".format(AP))
    f.close()
    print(AP_ans)



run_model()
