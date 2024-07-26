import argparse
import time
from datetime import datetime
import cv2 as cv
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')

  args = parser.parse_args()

  ext_delegate = None
  ext_delegate_options = {}

  # parse external delegate options
  if args.ext_delegate_options is not None:
    options = args.ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if (len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()
      else:
        raise RuntimeError('Error parsing delegate option: ' + o)

  # load external delegate
  if args.ext_delegate is not None:
    print('Loading external delegate from {} with args: {}'.format(
        args.ext_delegate, ext_delegate_options))
    ext_delegate = [
        tflite.load_delegate(args.ext_delegate, ext_delegate_options)
    ]

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=ext_delegate,
      num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32
  
  print("Floating model:", floating_model)

  # NxHxWxC, H:1, W:2
  model_height = input_details[0]['shape'][1]
  model_width = input_details[0]['shape'][2]
  print(f"Model Width: {model_width}, Image Height: {model_height}")
 
  # Initialize the video capture device
  cap = cv.VideoCapture(3)
  cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
  cap.set(cv.CAP_PROP_FPS, 30)
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

  # Check if the video capture device is opened successfully
  if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

  # Ignore the first invoke
  startTime = time.time()
  delta = time.time() - startTime
  print("Warm-up time:", '%.1f' % (delta * 1000), "ms\n")

  for a in range(5000000):
    startTime = time.time()
    
    ret, img = cap.read()
    if not ret:
      print("Error: Failed to capture image.")
      continue

    newTime = time.time()
    delta = time.time() - newTime
    print("Capture time:", '%.1f' % (delta * 1000), "ms\n")
 
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    if img is None or img.size == 0:
      print("Failed to load image.")
      continue
    else:
      print(f"Captured Image Shape: {img.shape}, Dtype: {img.dtype}")
      cv.imwrite('images/image' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + '.png', img)

    delta = time.time() - newTime
    newTime = time.time()
    print("Write time:", '%.1f' % (delta * 1000), "ms\n")

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    newTime = time.time()
    delta = time.time() - newTime
    print("CVTColor time:", '%.1f' % (delta * 1000), "ms\n")

    pil_img = Image.fromarray(img_rgb)

    delta = time.time() - newTime
    newTime = time.time()
    print("fromarray time:", '%.1f' % (delta * 1000), "ms\n")
    
    print(f"PIL Image Size: {pil_img.size}")  # Print the size of the PIL image

    # Resize image to W1 and H1 using PIL
    pil_resized_img = pil_img.resize((model_width, model_height))

    delta = time.time() - newTime
    newTime = time.time()
    print("Resized time:", '%.1f' % (delta * 1000), "ms\n")
        
    # Add N dim
    input_data = np.expand_dims(pil_resized_img, axis=0)
    delta = time.time() - newTime
    print("SetTensor time:", '%.1f' % (delta * 1000), "ms\n")
    
    #interpreter.invoke()
    delta = time.time() - newTime
    newTime = time.time()
    print("Interpreter invoke time:", '%.1f' % (delta * 1000), "ms\n")

    time.sleep(0.53)

    delta = time.time() - newTime
    newTime = time.time()
    print("Sleep time:", '%.1f' % (delta * 1000), "ms\n")
  
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
  
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
      if floating_model:
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      else:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
