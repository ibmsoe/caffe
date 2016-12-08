#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--labels_file",
        default=os.path.join(pycaffe_dir,
                "../data/ilsvrc12/synset_words.txt"),
        help="Readable label definition file."
    )
    parser.add_argument(
        "--print_results",
        action='store_true',
        help="Write output text to stdout rather than serializing to a file."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    else: 
        # channel-wise mean 
        mean = np.array([104,117,123])

    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        print("Loading file: %s" % args.input_file)
        inputs = [caffe.io.load_image(args.input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    scores = classifier.predict(inputs, not args.center_only).flatten()
    print("Done in %.2f s." % (time.time() - start))

#   The script has been updated to support --print_results option. 
#   Ref - http://stackoverflow.com/questions/37265197/classify-py-is-not-taking-argument-print-results
#   However, the labels format supported here has been modified, such that the file can have shorttext 
#   corresponding to the category classes instead of the general format. 
#   The commented part correspond to the general format of labels file which has mapping between 
#   synset_id and the text. 

    if args.print_results:
#        with open(args.labels_file) as f:
#          labels_df = pd.DataFrame([
#               {
#                   'synset_id': l.strip().split(' ')[0],
#                   'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
#               }
#               for l in f.readlines()
#            ])
#        labels_df.synset_id = labels_df.synset_id.astype(np.int64)
#        labels = labels_df.sort('synset_id')['name'].values

        labels_file = open(args.labels_file, 'r')
        labels = labels_file.readlines()
        
        indices = (-scores).argsort()[:5]
#        predictions = labels[indices]
#        print(predictions)
#        meta = [
#                   (p, '%.5f' % scores[i])
#                   for i, p in zip(indices, predictions)
#               ]
#        print meta
        print("---------------------------------")    
        print("The top 5 predictions are")
        for i in indices:
           print('%.4f %s' % (scores[i] , labels[i].strip('\n')))
        print("---------------------------------")

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, scores)


if __name__ == '__main__':
    main(sys.argv)
