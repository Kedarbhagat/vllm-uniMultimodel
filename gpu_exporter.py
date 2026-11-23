from prometheus_client import start_http_server, Gauge
import subprocess
import time

# Define Prometheus metrics
gpu_temperature = Gauge('nvidia_gpu_temperature_celsius', 'GPU Temperature in Celsius', ['gpu'])
gpu_utilization = Gauge('nvidia_gpu_utilization_percent', 'GPU Utilization Percentage', ['gpu'])
gpu_memory_used = Gauge('nvidia_gpu_memory_used_bytes', 'GPU Memory Used in Bytes', ['gpu'])
gpu_memory_total = Gauge('nvidia_gpu_memory_total_bytes', 'GPU Total Memory in Bytes', ['gpu'])
gpu_power = Gauge('nvidia_gpu_power_watts', 'GPU Power Usage in Watts', ['gpu'])

def get_gpu_stats():
    # Run nvidia-smi command to get GPU stats
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
         '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    output = result.stdout.decode('utf-8').strip().split('\n')
    for line in output:
        idx, temp, util, mem_used, mem_total, power = line.split(',')
        gpu_temperature.labels(idx.strip()).set(float(temp))
        gpu_utilization.labels(idx.strip()).set(float(util))
        gpu_memory_used.labels(idx.strip()).set(float(mem_used) * 1024 * 1024)  # MB to Bytes
        gpu_memory_total.labels(idx.strip()).set(float(mem_total) * 1024 * 1024)
        gpu_power.labels(idx.strip()).set(float(power))

if __name__ == '__main__':
    # Start Prometheus metrics server on port 9400
    start_http_server(9183)
    print("NVIDIA GPU Exporter running on port 9183...")
    while True:
        get_gpu_stats()
        time.sleep(5)
