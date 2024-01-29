import socket
import time
import subprocess


def send_data_to_netcat(host, port, items):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # s.sendall(b"Hello, Spark Streaming!")
        for item in items:
            data = f"{item[0]}\n"
            command = f"echo '{data}' | nc {host} {port}"
            subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            # s.sendall(data.encode("utf-8"))
            time.sleep(1)
    print("Finished sending data")


if __name__ == "__main__":
    netcat_host = "localhost"
    netcat_port = 1234
    num_producers = 8

    items = [(item,) for item in range(num_producers)]

    send_data_to_netcat(netcat_host, netcat_port, items)
