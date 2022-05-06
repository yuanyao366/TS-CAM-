
CUDA_VISIBLE_DEVICES=0 python ./tools_cam/test_cam_w2cls_2model.py \
--config_file configs/ILSVRC/deit_cls_attn_cam_small_patch16_224.yaml \
--resume ckpt/ImageNet/deit_cls_attn_cam_small_patch16_224_CAM-NORMAL_SEED26_SUM-CAM-THR0.12_BS64/LogitPT_LossCE1.0_2021-09-04-11-52/ckpt/model_epoch20.pth \
MODEL.CAM_THR 0.12 MODEL.CAM_MERGE SUM BASIC.BACKUP_CODES False TEST.BATCH_SIZE 64 TEST.CAM_TYPE MeanAttenCam TEST.CLS_LOGITS ImageNet TEST.IOU_THR 0.5