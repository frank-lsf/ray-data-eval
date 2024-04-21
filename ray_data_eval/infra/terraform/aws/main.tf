provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "cluster" {
  for_each = toset(var.instances)

  instance_type               = each.value
  ami                         = "ami-079e23e92981f0030"
  key_name                    = "login-us-west-2"
  subnet_id                   = "subnet-033a4bd05bcc8c21e" # exoshuffle-subnet-public3-us-west-2c
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.instance_disk_gb
  }

  tags = {
    ClusterName = var.cluster_name
    Name        = "${var.cluster_name}-${each.value}"
  }
}