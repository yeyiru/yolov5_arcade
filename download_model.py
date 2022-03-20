import os
from detect import main as detect_main
import wandb
import argparse
import shutil
import pandas as pd

run = wandb.init(project='yolov5_s_test')

# ---------------------------coco_small_weights-----------------------------
coco_small_weights = ['run_2fb6aj8e_model:v0',
                      'run_7uqunfqp_model:v0',
                      'run_2xro22qo_model:v0',
                      'run_yjzh0ri6_model:v0',
                      'run_pt5ermv8_model:v0']
coco_small_iou = [0.6701, 0.1778, 0.3820, 0.2475, 0.2631]

# ---------------------------coco_merge_weights-----------------------------
coco_merge_weights = ['run_qnkq2woh_model:v0',
                      'run_us7mj8cc_model:v0',
                      'run_q48un0e1_model:v0',
                      'run_8iyelcff_model:v0',
                      'run_9dsd2qy0_model:v0']
coco_merge_iou = [0.2818, 0.6963, 0.4127, 0.2002, 0.1651]

# # ---------------------------beauty_small_weights-----------------------------
# beauty_small_weights = ['run_xzafqu5a_model:v0',
#                         'run_o4518e8o_model:v0',
#                         'run_5enzr0ef_model:v0',
#                         'run_r4i78c5z_model:v0',
#                         'run_muiin0mw_model:v0']
# beauty_small_iou = [0.5818, 0.5141, 0.5204, 0.5563, 0.6837]
# # ---------------------------beauty_merge_weights-----------------------------
# beauty_merge_weights = ['run_3vx55hc7_model:v0',
#                         'run_vwggzein_model:v0',
#                         'run_grep74lq_model:v0',
#                         'run_avfyycoi_model:v0',
#                         'run_k42p8aqv_model:v0']
# beauty_merge_iou = [0.2119, 0.658, 0.1177, 0.1169, 0.214]


# ------------------------nopre_small_weights--------------------------
nopre_small_weights = ['run_5zop1tvy_model:v0',
                       'run_weit0uf9_model:v0',
                       'run_cgiba9r4_model:v0',
                       'run_sthowv71_model:v0',
                       'run_q8qv8hdn_model:v0']
nopre_small_iou = [0.6001, 0.6696, 0.1976, 0.3012, 0.2188]
# ------------------------nopre_merge_weights--------------------------
nopre_merge_weights = ['run_kkomhia8_model:v0',
                       'run_g5j318wo_model:v0',
                       'run_2cpqxh1x_model:v0',
                       'run_2vyn9mie_model:v0',
                       'run_0aw26sdg_model:v0']
nopre_merge_iou = [0.4913, 0.2064, 0.6363, 0.5648, 0.4785]

def download_weights(mode, weight_idx):
    if 'small' in mode:
        if 'nopre' in mode:
            weight_name = nopre_small_weights[weight_idx]
            iou = nopre_small_iou[weight_idx]
        # elif 'beauty' in mode:
        #     weight_name = beauty_small_weights[weight_idx]
        #     iou = beauty_small_iou[weight_idx]
        else:
            weight_name = coco_small_weights[weight_idx]
            iou = coco_small_iou[weight_idx]

    elif 'merge' in mode:
        if 'nopre' in mode:
            weight_name = nopre_merge_weights[weight_idx]
            iou = nopre_merge_iou[weight_idx]
        # elif 'beauty' in mode:
        #     weight_name = beauty_merge_weights[weight_idx]
        #     iou = beauty_merge_iou[weight_idx]
        else:
            weight_name = coco_merge_weights[weight_idx]
            iou = coco_merge_iou[weight_idx]
        # source = '/data/Arcade_dataset/arcade_merge/images/test/'

    artifact = run.use_artifact(f'yeyiru19970825/YOLOv5_All/{weight_name}', type='model')
    
    artifact_dir = artifact.download()
    save_dir = './artifacts/s_' + mode
    try:
        os.mkdir(save_dir)
    except:
        pass
    model_name = os.listdir(artifact_dir)[0]

    shutil.copyfile(os.path.join(artifact_dir, model_name), os.path.join(save_dir, str(weight_idx) + '_' + str(iou) + '.pt'))
    shutil.rmtree(artifact_dir)
    
    print('Model path is ' + os.path.join(save_dir, str(weight_idx) + '.pt'))
    return 

# for mode1 in ['nopre', 'coco', 'beauty']:
for mode1 in ['nopre', 'coco']:
    for mode2 in ['small', 'merge']:
        mode = mode1 + '_' + mode2
        for i in range(0, 5):
            download_weights(mode, i)
