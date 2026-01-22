#!/bin/bash

USER_NAME=$(whoami)
PIDS=$(ps -ef | grep "$USER_NAME" | grep -E 'python|pytest' | grep -v grep | awk '{print $2}')

for PID in $PIDS; do
    kill -9 "$PID"
done
