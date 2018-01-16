#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic language modeling with RNN."""

import argparse
import json
import math
import os
import time

import numpy as np
# To get rid of stupid tensorflow logging. Works from version 0.12.1+.
# See https://github.com/tensorflow/tensorflow/issues/1258
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # noqa
import tensorflow as tf

import data
from tf_model import LSTMModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Modification of the PyTorch Wikitext-2 RNN/LSTM Language '
                    'Model, so that it actually does what Zaremba (2014) '
                    'described.')
    parser.add_argument('--data', '-d', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--configuration', '-c', type=str, default='config.json',
                        help='the configuration file (json).')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='the model key name.')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()

    try:
        config = read_config(args.configuration, args.model)
        return args, config
    except Exception as e:
        parser.error('Error reading configuration file: {}'.format(e))


class AttrDict(dict):
    """Makes our life easier."""
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        self[key] = value


def read_config(config_file, model):
    with open(config_file) as inf:
        return AttrDict(json.load(inf)[model])


def batchify(data, bsz):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, bptt, evaluation=False):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].numpy()
    target = source[i+1:i+1+seq_len].numpy()
    return data, target


# def run_epoch(session, model, data, epoch_size=0, verbose=0,
#               global_step=0, writer=None):
#     """
#     Runs an epoch on the network.
#     - epoch_size: if 0, it is taken from data
#     - data: a DataLoader instance
#     """
#     # TODO: these two should work together better, i.e. keep the previous
#     #       iteration state intact if epoch_size != 0; also loop around
#     epoch_size = data.epoch_size if epoch_size <= 0 else epoch_size
#     data_iter = iter(data)
#     start_time = time.time()
#     costs = 0.0
#     iters = 0
#     state = session.run(model.initial_state)
#
#     fetches = [model.cost, model.final_state, model.train_op]
#     fetches_summary = fetches + [model.summaries]
#     if verbose:
#         log_every = epoch_size // verbose
#
#     for step in range(epoch_size):
#         x, y = next(data_iter)
#
#         # feed_dict = {model.sequence: batch}
#         # for i, (c, h) in enumerate(model.initial_state):
#         #     feed_dict[c] = state[i].c  # CEC for layer i
#         #     feed_dict[h] = state[i].h  # hidden for layer i
#         feed_dict = {
#             model.input_data: x,
#             model.targets: y,
#             model.initial_state: state
#         }
#
#         if verbose and step % log_every == log_every - 1:
#             cost, state, _, summary = session.run(fetches_summary, feed_dict)
#             if writer:
#                 writer.add_summary(summary, global_step=global_step)
#             if model.is_training:
#                 global_step += 1
#         else:
#             cost, state, _ = session.run(fetches, feed_dict)
#         # logger.debug('Cost: {}'.format(cost))
#         costs += cost
#         iters += model.params.num_steps
#         if verbose and step % log_every == log_every - 1:
#             logger.debug(
#                 "%.3f perplexity: %.3f speed: %.0f wps" %
#                 (step * 1.0 / epoch_size, np.exp(costs / iters),
#                  iters * model.params.batch_size / (time.time() - start_time))
#             )
#
#     # global_step is what the user sees, i.e. if the output is verbose, it is
#     # increased, otherwise it isn't
#     if not verbose and model.is_training:
#         global_step += 1
#
#     return np.exp(costs / iters), global_step


def train(sess, model, corpus, train_data, epoch, lr, config, log_interval):
    def to_str(f):
        return corpus.dictionary.idx2word[f]

    # Turn on training mode which enables dropout.
    model.assign_lr(sess, lr)
    total_loss = 0
    start_time = time.time()
    fetches = [model.cost, model.predictions, model.final_state, model.train_op]
    hidden = sess.run(model.initial_state)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, config.bptt)):
        data, targets = get_batch(train_data, i, config.bptt)
        print('DATA\n', np.vectorize(to_str)(data))
        print('TARGETS\n', np.vectorize(to_str)(targets))

        feed_dict = {
            model.input_data: data,
            model.targets: targets,
            model.initial_state: hidden
        }
        cost, output, hidden, _ = sess.run(fetches, feed_dict)

        indices = np.argmax(output, 2)
        print('OUTPUT\n', np.vectorize(to_str)(indices.data.cpu().numpy()))

        total_loss += cost

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // config.bptt, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(sess, model, corpus, data_source, batch_size, bptt):
    total_loss = 0
    fetches = [model.cost, model.predictions, model.final_state]
    hidden = sess.run(model.initial_state)

    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        feed_dict = {
            model.input_data: data,
            model.targets: targets,
            model.initial_state: hidden
        }
        cost, output, hidden = sess.run(fetches, feed_dict)
        total_loss += cost
    print('TOTAL LOSS', total_loss, 'LEN DATA', len(data_source), data_source.size())
    return total_loss / len(data_source)


def main():
    args, config = parse_arguments()

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    train_batch_size = config.batch_size
    eval_batch_size = 10
    train_data = batchify(corpus.train, train_batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)
    eval_batch_size = 10
    ntoken = len(corpus.dictionary)

    # Create the models and the global ops
    with tf.Graph().as_default() as graph:
        tf.set_random_seed(args.seed)
        # init_scale = 1 / math.sqrt(args.num_nodes)
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.name_scope('Train'):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                mtrain = LSTMModel(True, ntoken, config.nhid, config.nlayers,
                                   config.batch_size, config.bptt, config.clip)
        with tf.name_scope('Valid'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = LSTMModel(False, ntoken, config.nhid, config.nlayers,
                                   eval_batch_size, config.bptt, config.clip)
        with tf.name_scope('Test'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = LSTMModel(False, ntoken, config.nhid, config.nlayers,
                                  eval_batch_size, config.bptt, config.clip)
        with tf.name_scope('Global_ops'):
            init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)

    ###############################################################################
    # Training code
    ###############################################################################

        # Loop over epochs.
        lr = config.lr

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, config.epochs + 1):
                lr_decay = config.lr_decay ** max(epoch - config.decay_delay, 0.0)
                lr = config.lr * lr_decay
                epoch_start_time = time.time()
                train(sess, mtrain, corpus, train_data,
                      epoch, lr, config, args.log_interval)
                val_loss = evaluate(sess, mvalid, corpus, val_data,
                                    eval_batch_size, config.bptt)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Run on test data.
        test_loss = evaluate(sess, mtest, corpus, test_data,
                             eval_batch_size, config.bptt)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main()
