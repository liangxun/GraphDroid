model_save_dir = "GraphDroid/baseline/save_models"
data_dir = "GraphDroid/data"

# ======= model config
feature_dim = 4075
layers_out = [200, 200, 2]

# ======= train config
epoch = 5
batchsize = 128
lr = 1e-3
drop_rt = 0.4

cuda_device_id = 3  # gpu device
