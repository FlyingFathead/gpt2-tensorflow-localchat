#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Refactored by FlyingFathead from Telegram Bot to a local CLI application
# Original fork: https://github.com/FlyingFathead/GPT2-Telegram-Chatbot
# Refactored version: Local CLI

import json, os, sys, threading, random, logging
import tensorflow as tf
import numpy as np
import model, sample, encoder

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and conversation settings
input_prefix = "|k| "
output_prefix = "|v| "
starting_context = "\n"
debug = True
top = 0.77
degree = 1.0
mx = 0.00500
tok = 0
learning = ""
mode = True
learn = True
cache = ""
turns = []

def interact_model(text_input):
    model_name = 'fenno'
    seed = random.randint(1, 10000)
    nsamples = 1
    batch_size = 1
    top_k = tok
    top_p = top
    models_dir = os.path.expanduser('~/NeuralNetwork/vzgpt/__incheck/')  # Updated directory

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=1024,  # Modify length according to your preference
            context=context,
            batch_size=batch_size,
            temperature=degree, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(input_prefix + text_input + '\n' + output_prefix)
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        text = enc.decode(out[0])
        return text.split('\n')[0]  # Return first line of output

def main_loop():
    print("Welcome to the CLI Chatbot. Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        response = interact_model(inp)
        print("Bot:", response)

if __name__ == '__main__':
    main_loop()
