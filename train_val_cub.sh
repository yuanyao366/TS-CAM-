
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools_cam/train_cam_w2cls.py \
--config_file ./configs/CUB/deit_cls_attn_cam_small_patch16_224.yaml \
--lr 5e-5 MODEL.CAM_THR 0.1 TEST.CLS_LOGITS PT