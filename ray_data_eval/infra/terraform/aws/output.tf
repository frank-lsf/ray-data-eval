output "instance_ips" {
  value = [for instance in aws_instance.cluster : instance.private_ip]
}

output "instance_ids" {
  value = [for instance in aws_instance.cluster : instance.id]
}

output "instance_types" {
  value = [for instance in aws_instance.cluster : instance.instance_type]
}
