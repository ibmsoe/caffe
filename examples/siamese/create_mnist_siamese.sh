#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

DATA=./data/mnist

# Check if CAFFE_NV_BIN is set
if [ -z ${CAFFE_NV_BIN+x} ];
# if unset
then
  EXAMPLES=./build/examples/siamese
else
  EXAMPLES=$CAFFE_NV_BIN
fi

echo "Creating leveldb..."

rm -rf ./examples/siamese/mnist_siamese_train_leveldb
rm -rf ./examples/siamese/mnist_siamese_test_leveldb

$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_train_leveldb
$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_test_leveldb

echo "Done."
