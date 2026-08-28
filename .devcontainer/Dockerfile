# Use the 2026-standard slim Python image
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# Set the environment variable for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Base image bundles a yarn apt source with a stale/unreachable signing key.
# Not needed for this lab — remove it before updating package lists.
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Update system and install essential enterprise networking tools
RUN apt-get update && apt-get install -y \
curl \
git \
openssh-client \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the workspace directory
WORKDIR /workspace