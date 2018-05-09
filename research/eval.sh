python3 object_detection/eval.py \
  --logtostderr \
  --pipeline_config_path=object_detection/waste_busters/faster_rcnn_resnet101_cups.config \
  --checkpoint_dir=object_detection/waste_busters/train/ \
  --eval_dir=object_detection/waste_busters/eval/