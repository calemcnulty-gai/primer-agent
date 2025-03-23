#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="calemcnulty"
AGENT_NAME="primer-agent"
SECRET_SET="primer-agent-secrets"
IMAGE_TAG="0.1"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${AGENT_NAME}:${IMAGE_TAG}"

# Print banner
echo "====================================================="
echo "    Deploying ${AGENT_NAME} to Pipecat Cloud         "
echo "====================================================="

# Build the Docker image (targeting ARM architecture for cloud deployment)
echo "[1/4] Building Docker image..."
docker build --platform=linux/arm64 -t ${AGENT_NAME}:latest .

# Tag the image with Docker username and version
echo "[2/4] Tagging Docker image as ${FULL_IMAGE_NAME}..."
docker tag ${AGENT_NAME}:latest ${FULL_IMAGE_NAME}

# Push to Docker Hub
echo "[3/4] Pushing image to Docker Hub..."
docker push ${FULL_IMAGE_NAME}

# Deploy to Pipecat Cloud with force flag
echo "[4/4] Deploying agent to Pipecat Cloud..."
pcc deploy ${AGENT_NAME} ${FULL_IMAGE_NAME} --secrets ${SECRET_SET} --min-instances 1 --max-instances 1 --force

# Start the agent with force flag
pcc agent start ${AGENT_NAME} --use-daily --force

echo "====================================================="
echo "                Deployment Complete                  "
echo "====================================================="