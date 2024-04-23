variable "instances" {
  type    = list(string)
  default = ["g5.xlarge"]
}

variable "cluster_name" {
  default = "ray-data-eval"
}

variable "instance_disk_gb" {
  default = 200
}
