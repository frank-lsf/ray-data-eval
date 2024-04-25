variable "instances" {
  type = map(string)
  default = {
    "node_01" = "m7i.xlarge"
  }
}

variable "cluster_name" {
  default = "ray-data-eval"
}

variable "instance_disk_gb" {
  default = 200
}
