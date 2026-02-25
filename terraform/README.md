# Infrastructure as Code (Terraform)

This directory contains the Terraform configuration to provision the AWS infrastructure for the Academic Risk Model API.

## Prerequisites

1.  **Terraform**: Install Terraform (v1.0+).
2.  **AWS CLI**: Configured with valid credentials (`aws configure`).
3.  **SSH Key**: Ensure you have an SSH public key at `~/.ssh/id_rsa.pub`. If your key is elsewhere, update `terraform/main.tf`.

## Resources Provisioned

*   **EC2 Instance**: Amazon Linux 2023, t3.medium (configurable).
*   **Security Group**: Allows inbound traffic on ports 22 (SSH), 80/443 (HTTP/S), and 5000 (Flask API).
*   **IAM Role**: Uses existing `LabRole` (common in AWS Academy environments).

## Usage

1.  **Initialize Terraform**:
    ```bash
    cd terraform
    terraform init
    ```

2.  **Preview Changes**:
    ```bash
    terraform plan
    ```

3.  **Apply Changes**:
    ```bash
    terraform apply
    ```
    Confirm with `yes`.

4.  **Get Output**:
    After successful application, Terraform will output the `public_ip` and `api_url`.

## Application Deployment

Once the infrastructure is ready, deploy the application using the script in `scripts/deploy_aws.sh`.

```bash
# Get the Public IP from terraform output
export PUBLIC_IP=$(terraform output -raw public_ip)

# Run the deployment script on the remote server
ssh -i ~/.ssh/id_rsa ec2-user@$PUBLIC_IP 'bash -s' < ../scripts/deploy_aws.sh
```

## Continuous Deployment (GitHub Actions)

This project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically deploys the application to the EC2 instance whenever changes are pushed to `main`.

**Important**: The workflow assumes the infrastructure is **already running**.

### Setup
1.  Run `terraform apply` to create the instance.
2.  Copy the `public_ip` from the output.
3.  Add the `public_ip` as `EC2_HOST` in your GitHub Repository Secrets.

## Cleanup

To destroy the infrastructure:

```bash
terraform destroy
```
