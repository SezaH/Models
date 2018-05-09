#python3 ./TFR_converter.py --output_path=outfile.records
#python3 TFR_converter.py --output_path training.record
import tensorflow as tf
from xml.dom import minidom
import base64

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(i):
  # TODO: Populate the following variables from your example.
  height = 400 # Image height
  width = 600 # Image width
  #print(type(((b'0')+i))
  filename = b'image' + bytes(str(i),"ascii") + b'.jpeg' # Filename of the image. Empty if image is not from file
  #filename = b'image1.jpg'
  img_file = open(filename, 'rb')
  print(filename)
  encoded_image_data = img_file.read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [333/600] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [425/600] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [72/400] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [158/400] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [b'cup'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature( filename  ),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def main(_):

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO: Write code to read in your dataset to examples variable

  for i in range(0,10):
    tf_example = create_tf_example(i)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
