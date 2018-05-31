python3 object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path=./object_detection/waste_busters/faster_rcnn_resnet101_cups.config \
  --trained_checkpoint_prefix=./object_detection/waste_busters/train/model.ckpt-$1 \
  --output_directory ./object_detection/waste_busters/export/faster_rcnn_resnet101_cups_$1/

cd object_detection/waste_busters/export/

tar -zcvf faster_rcnn_resnet101_cups_$1.tar.gz \
  faster_rcnn_resnet101_cups_$1/*

mv faster_rcnn_resnet101_cups_$1/frozen_inference_graph.pb  faster_rcnn_resnet101_cups_$1.pb

rm -rf faster_rcnn_resnet101_cups_$1