#!/bin/bash
set -e

# Usage: ssh -i ~/.ssh/id_rsa ec2-user@<PUBLIC_IP> 'bash -s' < scripts/deploy_aws.sh

APP_NAME=academic-risk-model
APP_DIR=/home/ec2-user/$APP_NAME
# Replace with your actual repo URL if different
REPO_URL="https://github.com/ManoelSilva/academic-risk-model.git"
BRANCH=main

# Ensure docker is running
if ! systemctl is-active --quiet docker; then
    sudo systemctl start docker
fi

# Create app directory
if [ ! -d "$APP_DIR" ]; then
  mkdir -p $APP_DIR
fi

# Clone or pull
if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch $BRANCH $REPO_URL $APP_DIR
else
  cd $APP_DIR
  git pull origin $BRANCH
fi

cd $APP_DIR

# Build and start with Docker Compose
# Use sudo if user is not in docker group (though user_data adds ec2-user)
# Re-login is needed for group change to take effect in current shell, so we use sudo or newgrp
sudo docker compose down || true
sudo docker compose up -d --build

echo "Deployment completed. API should be running on port 5000."
