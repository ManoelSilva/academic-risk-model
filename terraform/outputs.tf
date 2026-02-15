output "public_ip" {
  value = aws_instance.academic_risk_host.public_ip
}

output "api_url" {
  value = "http://${aws_instance.academic_risk_host.public_ip}:5000"
}
