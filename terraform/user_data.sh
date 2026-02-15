#!/bin/bash
# Log all output to a file for troubleshooting
exec > >(tee /var/log/user_data.log|logger -t user-data -s 2>/dev/console) 2>&1
set -e

# Update and install system packages (Amazon Linux 2023 uses dnf)
echo "Checking for Amazon Linux 2023 release updates..."
dnf check-release-update || true

echo "Applying Amazon Linux 2023 release update if available..."
dnf upgrade -y || true

echo "Updating system packages..."
dnf update -y

echo "Installing Python 3.12 and pip..."
dnf install -y python3.12 python3.12-pip || {
    echo "python3.12 not found in default repos, falling back to python3"
    dnf install -y python3 python3-pip
}

echo "Installing git, unzip, wget..."
dnf install -y git unzip wget

# Install Docker
echo "Installing Docker..."
dnf install -y docker
usermod -aG docker ec2-user
systemctl enable docker
systemctl start docker

# Install Docker Compose (V2 plugin)
echo "Installing Docker Compose..."
dnf install -y docker-compose-plugin

# Upgrade pip
echo "Upgrading pip..."
if command -v python3.12 &> /dev/null; then
    python3.12 -m pip install --upgrade pip
else
    python3 -m pip install --upgrade pip
fi

# Print installed versions for debugging
echo "Python version:"
python3.12 --version || python3 --version
echo "Pip version:"
pip3.12 --version || pip3 --version
git --version
docker --version
docker compose version

echo "User data script execution completed successfully."
