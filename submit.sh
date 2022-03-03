#!/bin/bash

# upgrade pip and pillow
pip install --upgrade pip
pip install --upgrade pillow

# install required python packages
pip install -r requirements.txt

# run the code
python3 src/main.py --config config/config.yml