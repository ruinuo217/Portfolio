#!/bin/bash
source .venv/bin/activate
pkill gunicorn
gunicorn api.main:app --bind 127.0.0.1:3939 --daemon