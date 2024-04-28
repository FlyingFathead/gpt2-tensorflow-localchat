#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Model-Localtalk.py
#
# Refactored by FlyingFathead from Telegram Bot to a local CLI application
# Original fork: https://github.com/FlyingFathead/GPT2-Telegram-Chatbot
# Refactored version: Local CLI

import tensorflow as tf
import os
import json
import sys
import threading
import random
import logging
import numpy as np
import model
import sample
import encoder

# Set the GPU you want to use here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust this as needed to switch GPUs

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Suppresses info and warning messages

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)
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

class ModelWrapper:
    def __init__(self, models_dir, model_name=''):
        self.models_dir = models_dir
        self.model_name = model_name  # this could be an empty string if not using subdirectories
        self.turns = []  # Initialize turns inside the class        
        self.load_model()

    def load_model(self):
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        with self.session.graph.as_default():
            self.enc = encoder.get_encoder(self.model_name, self.models_dir)  # Adjusted here
            self.hparams = model.default_hparams()
            with open(os.path.join(self.models_dir, 'hparams.json')) as f:
                self.hparams.override_from_dict(json.load(f))
            self.context = tf.compat.v1.placeholder(tf.int32, [1, None])
            self.output = sample.sample_sequence(
                hparams=self.hparams, length=1000, # hold ~24 tokens under 1000
                context=self.context,
                batch_size=1,
                temperature=degree, top_k=tok, top_p=top
            )
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.models_dir)
            saver.restore(self.session, ckpt)

    def decode(self, tokens):
        try:
            text = ''.join([self.enc.decoder[token] if token in self.enc.decoder else '[UNK]' for token in tokens])
        except KeyError as e:
            print(f"Unknown token ID {e.args[0]} encountered in the output.")
            text = "[Decode Error]"
        return text

    def interact_model(self, text_input):
        # Encode the new input and append it to the dialogue history
        new_input_tokens = self.enc.encode(input_prefix + text_input + '\n' + output_prefix)
        self.turns.append(new_input_tokens)
        
        # Flatten the list of token lists to a single list of tokens
        context_tokens = [token for sublist in self.turns for token in sublist]
        
        # Trim tokens from the beginning if exceeding the max context length
        max_context_length = 1000  # Adjust based on model capabilities; 1024 usual max, keep a few tokens spare.
        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[-max_context_length:]
        
        # Process the context
        out = self.session.run(self.output, feed_dict={self.context: [context_tokens]})[:, len(context_tokens):]
        
        try:
            text = self.enc.decode(out[0])
        except KeyError as e:
            print(f"Unknown token ID {e.args[0]} encountered in the output.")
            text = "[Decode Error]"
        
        return text.split('\n')[0]

# Usage
model_wrapper = ModelWrapper(os.path.expanduser('~/NeuralNetwork/vzgpt/__incheck/'))

def main_loop():
    print("Welcome to the CLI Chatbot. Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        response = model_wrapper.interact_model(inp)
        print("Bot:", response)

if __name__ == '__main__':
    main_loop()