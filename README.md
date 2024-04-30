# gpt2-tensorflow-localchat

`gpt2-tensorflow-localchat` is a simple CLI chat mode framework for Python, built for locally running GPT-2 models with TensorFlow. This tool provides an easy way to interact with GPT-2 models and fine-tune them on custom data sets or use them for unique, real-time applications.

## Project Repository
Explore more about this project and its developments on GitHub: [gpt2-tensorflow-localchat](https://github.com/FlyingFathead/gpt2-tensorflow-localchat)

## Features
- CLI-based interaction with GPT-2 models.
- Local deployment of TensorFlow models for privacy and control.
- Support for multiple command scripts to demonstrate various capabilities.

## Directory Structure
```
├── .gitignore
├── README.md
└── src/
    ├── Model-Battle.py          # Battle between models: experimental feature
    ├── Model-Localtalk.py       # Main script for local chat interactions
    ├── encoder.py               # Manages text encoding and decoding
    ├── model.py                 # Core TensorFlow model definitions
    ├── olddemo.py               # Old demonstration scripts for reference
    ├── sample.py                # Sampling utilities for generating text
    └── start_localtalk.sh       # Script to start local chat environment(*)
```
_(*) The purpose of the bash script is to suppress Tensorflow's output from interfering in the CLI output._

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow 1.15 or higher (with `compat.v1` API support)
- An environment supporting bash scripts (Linux/Unix)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FlyingFathead/gpt2-tensorflow-localchat.git
   ```
2. Navigate into the project directory:
   ```bash
   cd gpt2-tensorflow-localchat
   ```
3. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
_(Note: `requirements.txt` currently not added in yet. Use your own local TF model files.)_

### Usage
To start a local chat with the model:
```bash
./src/start_localtalk.sh
```
This script sets the appropriate TensorFlow logging level and starts an interactive chat session using `Model-Localtalk.py`.

## Changes
- `v0.16` - `/clear` to clear out the context memory
- `v0.15` - bugfixes, `/swap` for role-swapping between user and the model
- `v0.10` - initial commit

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## License
Distributed under the MIT License. See `LICENSE` for more information. Parts of the model loading code has been forked from [OpenAI's GPT-2 source code](https://github.com/openai/gpt-2).

## Contact
- Project Link: [https://github.com/FlyingFathead/gpt2-tensorflow-localchat](https://github.com/FlyingFathead/gpt2-tensorflow-localchat)
- Project Creator: [Flyingfathead on GitHub](https://github.com/FlyingFathead/)