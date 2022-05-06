
CUDA_VISIBLE_DEVICES=0 python ./tools_cam/test_cam_w2cls_2model.py \
--config_file configs/CUB/deit_cls_attn_cam_small_patch16_224.yaml \
--resume ckpt/CUB/deit_cls_attn_cam_small_patch16_224_CAM-NORMAL_SEED26_SUM-CAM-THR0.1_BS128/LogitPT_LossCE1.0_2021-08-30-12-29/ckpt/model_epoch60.pth \
--resume_cls ckpt/CUB/deit_small_patch16_224_CAM-NORMAL_SEED26_BS128/2021-09-09-13-12/ckpt/model_best_top1_acc.pth \
MODEL.CAM_THR 0.1 MODEL.CAM_MERGE SUM BASIC.BACKUP_CODES False TEST.BATCH_SIZE 128 TEST.CAM_TYPE MeanAttenCam TEST.CLS_LOGITS ImageNet TEST.IOU_THR 0.5
