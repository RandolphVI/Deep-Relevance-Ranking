# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


class TextDRMM(object):
    """A DRMM for text classification."""

    def __init__(
            self, sequence_length, num_classes_list, total_classes, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y_first = tf.placeholder(tf.float32, [None, num_classes_list[0]], name="input_y_first")
        self.input_y_second = tf.placeholder(tf.float32, [None, num_classes_list[1]], name="input_y_second")
        self.input_y_third = tf.placeholder(tf.float32, [None, num_classes_list[2]], name="input_y_third")
        self.input_y = tf.placeholder(tf.float32, [None, total_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)

            # Average Vectors
            # [batch_size, embedding_size]
            self.embedded_sentence_average = tf.reduce_mean(self.embedded_sentence, axis=1)

        # First Level
        with tf.name_scope("first-fc"):
            W = tf.Variable(tf.truncated_normal(shape=[embedding_size, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.first_fc = tf.nn.xw_plus_b(self.embedded_sentence_average, W, b)
            self.first_fc_out = tf.nn.relu(self.first_fc, name="relu")

        with tf.name_scope("first-output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes_list[0]],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes_list[0]], dtype=tf.float32), name="b")
            self.first_logits = tf.nn.xw_plus_b(self.first_fc_out, W, b, name="logits")
            self.first_scores = tf.sigmoid(self.first_logits, name="scores")

        # Second Level
        with tf.name_scope("second-fc"):
            self.second_input = tf.concat([self.first_scores, self.embedded_sentence_average], axis=1)
            W = tf.Variable(tf.truncated_normal(shape=[(num_classes_list[0] + embedding_size), fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.second_fc = tf.nn.xw_plus_b(self.second_input, W, b)
            self.second_fc_out = tf.nn.relu(self.second_fc, name="relu")

        with tf.name_scope("second-output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes_list[1]],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes_list[1]], dtype=tf.float32), name="b")
            self.second_logits = tf.nn.xw_plus_b(self.second_fc_out, W, b, name="logits")
            self.second_scores = tf.sigmoid(self.second_logits, name="scores")

        # Third Level
        with tf.name_scope("third-fc"):
            self.third_input = tf.concat([self.second_scores, self.embedded_sentence_average], axis=1)
            W = tf.Variable(tf.truncated_normal(shape=[(num_classes_list[1] + embedding_size), fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.third_fc = tf.nn.xw_plus_b(self.third_input, W, b)
            self.third_fc_out = tf.nn.relu(self.third_fc, name="relu")

        with tf.name_scope("third-output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes_list[2]],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes_list[2]], dtype=tf.float32), name="b")
            self.third_logits = tf.nn.xw_plus_b(self.third_fc_out, W, b, name="logits")
            self.third_scores = tf.sigmoid(self.third_logits, name="scores")

        # Final scores
        with tf.name_scope("output"):
            self.scores = tf.concat([self.first_scores, self.second_scores, self.third_scores], axis=1, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            def cal_loss(labels, logits, name):
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name=name + "sigmoid_losses")
                return losses

            # Loss
            losses_1 = cal_loss(labels=self.input_y_first, logits=self.first_logits, name="first_")
            losses_2 = cal_loss(labels=self.input_y_second, logits=self.second_logits, name="second_")
            losses_3 = cal_loss(labels=self.input_y_third, logits=self.third_logits, name="third_")
            losses = tf.add_n([losses_1, losses_2, losses_3], name="losses")

            # L2 Loss
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add_n([losses, l2_losses], name="loss")
