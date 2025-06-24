#!/bin/bash

for arg in "$@"; do
  case $arg in
    proxy_port*)
      proxy_port="${arg#*=}"
      ;;        
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

export PROXY_PORT="${proxy_port:-8000}"

# run proxy
fastapi run server.py --port $PROXY_PORT &
# cache pid for killing in future
PROXY_PID=$!
echo "PROXY PID: $PROXY_PID"
echo $PROXY_PID > proxy.pid

# wait till proxy server up
echo "Waiting proxy server to start..."
for i in {1..240}; do
    if curl -s http://localhost:$PROXY_PORT/ping | grep -q '"pong"'; then
        echo "proxy server is up!"
        break
    fi
    echo "⏳ proxy server not ready yet... retrying ($i)"
    sleep 1
done
