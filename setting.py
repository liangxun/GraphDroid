import logging.config
import os


# ========================= logger ================================
log_conf = './logger.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('GraphDroid')


# ============================ files ==============================
script_root = os.path.abspath(__file__) # root_path.
data_dir = os.path.join(script_root, 'data')


# 结果保存的地址
model_path = os.path.join(script_root, "save_models")
embed_path = os.path.join(script_root, "visualation/embeddings")
report_file = os.path.join(script_root,"save_models","reports.csv")

cuda_device_id = 2