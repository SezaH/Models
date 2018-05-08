python3 object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path=./object_detection/models/cup/faster_rcnn_resnet101_cups.config \
  --trained_checkpoint_prefix=./object_detection/models/cup/train/model.ckpt-2683 \
  --output_directory ./object_detection/models/cup/export/