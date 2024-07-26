SE_GPU_INFERENCE=0 sudo python label_image.py -e /lib/libvx_delegate.so -m mobilenet_v1_1.0_224_quant.tflite -i grace_hopper.bmp -l labels.txt
