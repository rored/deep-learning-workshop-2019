import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from inception_transfer_learning.utils.inception_properties import Props
from inception_transfer_learning.utils.model import Model
from inception_transfer_learning.utils.utils import get_model_info, add_jpeg_decoding, create_model_graph
from mnist_training.mouse_conv_keras import load_dataset

CLASS_NUMBER = 3

model_path = '../../online_inception/data/inception_dir/classify_image_graph_def.pb'
model_info = get_model_info()
props = Props()

graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info, model_path)

accuracy = 0.0
with tf.Session(graph=graph) as sess:
    retrain = Model(learning_rate=props.learning_rate)

    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = retrain.add_final_training_ops(
        CLASS_NUMBER, props.final_tensor_name, bottleneck_tensor,
        model_info['bottleneck_tensor_size'], model_info['quantize_layer'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = retrain.add_evaluation_step(final_tensor, ground_truth_input)

    # Create all Summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(props.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        props.summaries_dir + '/validation')

    final_tensor = tf.get_default_graph().get_tensor_by_name('final_result:0')

    init = tf.global_variables_initializer()
    sess.run(init)

    x, y = load_dataset()
    x_tensor = []
    for img in x:
        img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), -1)
        tensor = retrain.run_bottleneck_on_image(sess, img, jpeg_data_tensor, decoded_image_tensor,
                                                 resized_image_tensor, bottleneck_tensor)
        x_tensor.append(tensor)

    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y, test_size=0.1)
    del x
    del y
    trainset = zip(x_train, y_train)
    testset = zip(x_test, y_test)

    i = 0
    for epoch in range(128):
        i += 1
        for batch in range(0, x_train.shape[0], 32):
            x, y = list(zip(*random.sample(trainset, 32)))
            train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: x, ground_truth_input: y})
            train_writer.add_summary(train_summary, i)

            # show results
            if (i % 50) == 0:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],
                                                               feed_dict={bottleneck_input: x,
                                                                          ground_truth_input: y})
                print('train accuracy:', train_accuracy)
