#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Model-Localtalk.py
#
# Refactored by FlyingFathead from Telegram Bot to a local CLI application
# Original fork: https://github.com/FlyingFathead/GPT2-Telegram-Chatbot
# Refactored version: Local CLI

version_number = 0.17

import tensorflow as tf
import re
import os
import json
import sys
import threading
import random
import logging
import numpy as np
import GPUtil
import model
import sample
import encoder

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the directory for models here
models_directory = os.path.expanduser('~/NeuralNetwork/vzgpt/__incheck/')

# Check for available GPUs using GPUtil
def get_best_gpu():
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPUs found")

    # Log detected GPUs
    for gpu in gpus:
        logging.info(f"GPU {gpu.id}: {gpu.name}, Free Memory: {gpu.memoryFree}MB, Total Memory: {gpu.memoryTotal}MB")

    # Sort GPUs by the free memory available
    best_gpu = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)[0]
    logging.info(f"Selected GPU {best_gpu.id}: {best_gpu.name} with {best_gpu.memoryFree}MB free memory")
    return best_gpu.id

# Set the GPU you want to use here
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust this as needed to switch GPUs

# Function to set the best GPU as the visible device
def set_best_gpu():
    use_best_gpu = True  # Set this flag based on your preference
    if use_best_gpu:
        best_gpu_id = get_best_gpu()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust this as needed to switch GPUs

# Call the function to set the best GPU
set_best_gpu()

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Suppresses info and warning messages

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
        self.max_context_length = 900  # Set your max context length here        
        self.turns = []  # Initialize turns inside the class        
        self.load_model()

        # Initialize prefixes
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix

        self.is_user_bot = False  # Track whether the user is acting as the bot

    def reset_context(self):
        # Reset the turns and any other necessary states
        self.turns = []
        print("Context cleared. Starting a new session.")

    def load_model(self):
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        with self.session.graph.as_default():
            self.enc = encoder.get_encoder(self.model_name, self.models_dir)  # Adjusted here
            self.hparams = model.default_hparams()
            with open(os.path.join(self.models_dir, 'hparams.json')) as f:
                self.hparams.override_from_dict(json.load(f))
            self.context = tf.compat.v1.placeholder(tf.int32, [1, None])
            self.output = sample.sample_sequence(
                hparams=self.hparams, length=900, # keep under 1024 tokens to avoid OOM garble
                context=self.context,
                batch_size=1,
                temperature=degree, top_k=tok, top_p=top
            )
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.models_dir)
            saver.restore(self.session, ckpt)
            # Print the checkpoint path after loading
            print(f"Loaded model checkpoint: {ckpt}")

    def swap_prefixes(self):
        self.input_prefix, self.output_prefix = self.output_prefix, self.input_prefix
        self.is_user_bot = not self.is_user_bot
        print("Swapped roles. User is now the bot." if self.is_user_bot else "Swapped back. User is the user.")

    def manage_tokens(self, tokens):
        # Decode tokens to text to analyze context
        text = self.decode(tokens)
        dialogue_blocks = text.split('\n')

        # Keep as many of the recent dialogue blocks as possible within the token limit
        new_dialogue = []
        current_length = 0
        
        # Iterate over the dialogue blocks from the end to the beginning
        for block in reversed(dialogue_blocks):
            encoded_block = self.enc.encode(block)
            if current_length + len(encoded_block) <= self.max_context_length:
                new_dialogue.append(block)
                current_length += len(encoded_block)
            else:
                break

        # Ensure there's at least one block pair to maintain dialogue continuity
        if len(new_dialogue) < 2 and len(dialogue_blocks) >= 2:
            new_dialogue = dialogue_blocks[-2:]

        # Since new_dialogue was built from the end backwards, reverse it to restore order
        new_dialogue.reverse()
        # Join the trimmed dialogue blocks back together with newlines
        reduced_text = '\n'.join(new_dialogue)
        tokens = self.enc.encode(reduced_text)
        
        # Log the new token length
        logging.debug(f"Managed tokens length: {len(tokens)}")
        return tokens


    # def manage_tokens(self, tokens):
    #     # Ensure the tokens list does not exceed max_context_length
    #     if len(tokens) > self.max_context_length:
    #         # Convert token IDs back to text to analyze context
    #         text = self.decode(tokens)
            
    #         # Split the text by newlines to separate input-output pairs
    #         dialogue_blocks = text.split('\n')
            
    #         # Keep as many of the recent dialogue blocks as possible within the token limit
    #         new_dialogue = []
    #         current_length = 0
            
    #         # Iterate over the dialogue blocks from the end to the beginning
    #         for block in reversed(dialogue_blocks):
    #             encoded_block = self.enc.encode(block)
    #             if current_length + len(encoded_block) <= self.max_context_length:
    #                 new_dialogue.append(block)
    #                 current_length += len(encoded_block)
    #             else:
    #                 break
            
    #         # Since new_dialogue was built from the end backwards, reverse it to restore order
    #         new_dialogue.reverse()
            
    #         # Join the trimmed dialogue blocks back together with newlines
    #         reduced_text = '\n'.join(new_dialogue)
    #         tokens = self.enc.encode(reduced_text)
            
    #     return tokens

    def decode(self, tokens):
        text = []
        for token in tokens:
            try:
                decoded_token = self.enc.decoder[token]
                text.append(decoded_token)
            except KeyError:
                # You could log the unknown token here if you want to keep track of how often this happens
                logging.error(f"Unknown token ID {token} encountered in the output.")
                # Optionally replace unknown token with a placeholder
                text.append('[UNK]')
        return ''.join(text)

    # def decode(self, tokens):
    #     try:
    #         text = ''.join([self.enc.decoder[token] if token in self.enc.decoder else '[UNK]' for token in tokens])
    #     except KeyError as e:
    #         print(f"Unknown token ID {e.args[0]} encountered in the output.")
    #         text = "[Decode Error]"
    #     return text

    def interact_model(self, text_input):
        new_input_tokens = self.enc.encode(self.input_prefix + text_input + '\n' + self.output_prefix)
        self.turns.append(new_input_tokens)

        # Flatten the list of token lists to a single list of tokens
        context_tokens = [token for sublist in self.turns for token in sublist]

        # Check if context_tokens is empty or nearly empty, which could break slicing
        if len(context_tokens) == 0:
            logging.error("No tokens available for interaction. Check the encoding process.")
            return "Error: No content to process."

        # Manage tokens to ensure we don't exceed the maximum length
        if len(context_tokens) > self.max_context_length:
            # More intelligent trimming can be applied here
            context_tokens = self.manage_tokens(context_tokens)

        # Safeguard to ensure context doesn't exceed max length and isn't empty
        if 0 < len(context_tokens) <= self.max_context_length:
            try:
                out = self.session.run(self.output, feed_dict={self.context: [context_tokens]})[:, len(context_tokens):]
            except Exception as e:
                logging.error(f"Failed during TensorFlow session run or output processing: {str(e)}")
                self.reset_context()
                return "[Execution Error]"

        else:
            # Handle cases where context_tokens length exceeds max_context_length or is zero
            if len(context_tokens) > self.max_context_length:
                context_tokens = self.manage_tokens(context_tokens)
                try:
                    out = self.session.run(self.output, feed_dict={self.context: [context_tokens]})[:, len(context_tokens):]
                except Exception as e:
                    logging.error(f"Failed during TensorFlow session run or output processing: {str(e)}")
                    self.reset_context()
                    return "[Execution Error]"
            else:
                logging.error("Unexpected token length: {}".format(len(context_tokens)))
                self.reset_context()
                return "Error: Context processing failure. Context has been reset."

        # Debugging: Log the number of tokens being processed
        if debug:
            logging.debug(f"Processing {len(context_tokens)} tokens.")

        # Attempt to run the model only when there are enough tokens to process
        try:
            out = self.session.run(self.output, feed_dict={self.context: [context_tokens]})
            result_tokens = out[:, len(context_tokens):]  # Slicing from the length of the input context
            if len(result_tokens[0]) == 0:
                logging.error("Zero token length detected in model output. Regenerating context.")
                self.reset_context()
                return "Error: Zero token length output. Context has been reset."
            text_output = self.enc.decode(result_tokens[0])
        except Exception as e:
            logging.error(f"Failed during TensorFlow session run or output processing: {str(e)}")
            return "[Execution Error]"

        return text_output.split('\n')[0]
 
# Usage
model_wrapper = ModelWrapper(models_directory)

def main_loop():
    print("Welcome to the CLI Chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("Bot: " if model_wrapper.is_user_bot else "You: ")
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == '/swap':
            model_wrapper.swap_prefixes()
        elif user_input.lower() == '/clear':
            model_wrapper.reset_context()
        else:
            response = model_wrapper.interact_model(user_input)
            print("You:" if model_wrapper.is_user_bot else "Bot:", response)
            
if __name__ == '__main__':
    main_loop()