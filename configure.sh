#!/bin/bash

pip install -r ./requirements_demo.txt
pip install "httpx[socks]"

rm -rf /home/featurize/work/.local/lib/python3.11/site-packages/socksio*

featurize port export 7860
