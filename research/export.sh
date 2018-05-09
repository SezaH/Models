python3 object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path=./object_detection/waste_busters/faster_rcnn_resnet101_cups.config \
  --trained_checkpoint_prefix=./object_detection/waste_busters/train/model.ckpt-$1 \
  --output_directory ./object_detection/waste_busters/export/faster_rcnn_resnet101_cups_$1/

tar -zcvf ./object_detection/waste_busters/export/faster_rcnn_resnet101_cups_$1.tar.gz \
  ./object_detection/waste_busters/export/faster_rcnn_resnet101_cups_$1/*

rm -r ./object_detection/waste_busters/export/faster_rcnn_resnet101_cups_$1