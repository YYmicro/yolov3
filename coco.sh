CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --batch 128 \
    --epochs 100 \
    --data coco.yaml \
    --weights yolov3.pt \
    --workers 32
