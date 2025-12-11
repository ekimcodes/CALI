import time
import requests
import json
import random

# Configuration
CLOUD_URL = "http://localhost:5000/upload_telemetry"
DEVICE_ID = "NeuroPatch-001"
BUFFER_SIZE = 5

class BaseStation:
    def __init__(self):
        self.buffer = []
        
    def listen_for_packets(self):
        """
        Simulate listening to BLE.
        In a real scenario, this uses 'bleak' or 'bluepy'.
        Here we will read from a 'ZMQ' socket or just function call if we run in same process, 
        but for 'simulation' of separate devices, we can just use a shared file or 
        have the Device Sim send HTTP requests TO the Base Station if we run Base Station as a server too?
        
        OR simpler: The Device Sim writes to a file, Base Station reads it (tail).
        OR Device Sim sends UDP packets.
        
        Let's use a shared file "ble_air_interface.jsonl" to mimic "Air".
        """
        try:
            # We'll read from stdin or a mock interface
            # For this prototype verification, let's just make the Base Station 
            # generate the request to Cloud, assuming it "received" it.
            # But the Device Sim is supposed to *send* it.
            
            # Better approach for Showcase:
            # Device Sim -> [Network/IPC] -> Base Station -> [HTTP] -> Cloud
            
            # Let's use UDP for "BLE" simulation.
            # Base Station listens on UDP port.
            pass
        except Exception as e:
            print(e)
            
    def process_packet(self, packet):
        print(f"[Hub] Received Packet: Seq {packet.get('seq')}")
        self.buffer.append(packet)
        
        if len(self.buffer) >= BUFFER_SIZE:
            self.upload_buffer()
            
    def upload_buffer(self):
        print(f"[Hub] Uploading {len(self.buffer)} packets to Cloud...")
        try:
            payload = {
                "device_id": DEVICE_ID,
                "packets": self.buffer
            }
            resp = requests.post(CLOUD_URL, json=payload)
            if resp.status_code == 200:
                print(f"[Hub] Cloud Ack: {resp.json()}")
                self.buffer = [] # Clear
            else:
                print(f"[Hub] Cloud Error: {resp.text}")
        except Exception as e:
            print(f"[Hub] Upload Failed: {e}")
            # Keep buffer (Retain for retry logic - omitted for prototype)

import socket

def run_base_station():
    # UDP Server to simulate BLE Receiver
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5555
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    bs = BaseStation()
    print(f"[Hub] Listening for BLE packets on {UDP_IP}:{UDP_PORT}...")
    
    while True:
        data, addr = sock.recvfrom(4096) # buffer size is 1024 bytes
        try:
            # Assume packet is JSON
            packet = json.loads(data.decode('utf-8'))
            bs.process_packet(packet)
        except Exception as e:
            print(f"[Hub] Corrupt packet: {e}")

if __name__ == "__main__":
    run_base_station()
