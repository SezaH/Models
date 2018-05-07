python3 object_detection/eval.py \
  --logtostderr \
  --pipeline_config_path=object_detection/models/cup/ssd_inception_v2_coco.config \
  --checkpoint_dir=object_detection/models/cup/train/ \
  --eval_dir=object_detection/models/cup/eval/