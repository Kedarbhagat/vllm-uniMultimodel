#!/usr/bin/env python3
"""
Windows Server Management Script for FastAPI Applications
Manages multiple FastAPI servers and workers from a single interface
"""

import subprocess
import threading
import time
import os
import sys
import signal
from typing import Dict, List, Optional
from dataclasses import dataclass
import psutil

@dataclass
class WindowsServer:
    name: str
    script_path: str
    working_dir: str
    command: Optional[str] = None
    port: Optional[int] = None
    use_uvicorn: bool = False

class WindowsServerManager:
    def __init__(self):
        self.servers: Dict[str, WindowsServer] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running: Dict[str, bool] = {}
        self.setup_servers()

    def setup_servers(self):
        """Setup server configurations"""
        self.servers = {
            "fastapi_gateway": WindowsServer(
                name="FastAPI Gateway",
                script_path="FASTAPI_GATEWAY.py",
                working_dir=r"C:\Users\STUDENT\Documents\vllm\REDIS_MODEL",
                use_uvicorn=False
            ),
            "worker_deepseek": WindowsServer(
                name="DeepSeek Worker",
                script_path="Worker_deepseek.py", 
                working_dir=r"C:\Users\STUDENT\Documents\vllm\REDIS_MODEL",
                use_uvicorn=False
            ),
            "worker_llama": WindowsServer(
                name="Llama Worker",
                script_path="Worker_llama.py",
                working_dir=r"C:\Users\STUDENT\Documents\vllm\REDIS_MODEL", 
                use_uvicorn=False
            ),
            "rag_server": WindowsServer(
                name="RAG Server",
                script_path="server.py",
                working_dir=r"C:\Users\STUDENT\Documents\vllm\RAG",
                command="uvicorn server:app --host 0.0.0.0 --port 9095 --workers 1 --log-level info",
                port=9095,
                use_uvicorn=True
            )
        }

    def build_command(self, server: WindowsServer) -> List[str]:
        """Build the command to execute"""
        if server.use_uvicorn and server.command:
            return server.command.split()
        else:
            return ["python", server.script_path]

    def start_server(self, name: str) -> bool:
        """Start a single server"""
        if name not in self.servers:
            print(f"❌ Server '{name}' not found in configuration")
            return False

        if name in self.processes and self.is_process_running(self.processes[name]):
            print(f"⚠️  Server '{name}' is already running")
            return True

        server = self.servers[name]
        print(f"🚀 Starting {server.name}...")
        
        try:
            # Check if working directory exists
            if not os.path.exists(server.working_dir):
                print(f"❌ Working directory does not exist: {server.working_dir}")
                return False

            # Check if script file exists
            script_full_path = os.path.join(server.working_dir, server.script_path)
            if not server.use_uvicorn and not os.path.exists(script_full_path):
                print(f"❌ Script file does not exist: {script_full_path}")
                return False

            cmd = self.build_command(server)
            
            # Create new console window for each server
            creation_flags = subprocess.CREATE_NEW_CONSOLE
            
            self.processes[name] = subprocess.Popen(
                cmd,
                cwd=server.working_dir,
                creationflags=creation_flags,
                shell=True
            )
            
            self.running[name] = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_server,
                args=(name,),
                daemon=True
            )
            monitor_thread.start()
            
            # Wait a moment to check if process started successfully
            time.sleep(2)
            if self.is_process_running(self.processes[name]):
                print(f"✅ {server.name} started successfully (PID: {self.processes[name].pid})")
                if server.port:
                    print(f"   📡 Server running on port {server.port}")
                return True
            else:
                print(f"❌ {server.name} failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start {server.name}: {e}")
            return False

    def stop_server(self, name: str):
        """Stop a single server"""
        if name not in self.processes:
            print(f"⚠️  Server '{name}' is not running")
            return

        server = self.servers[name]
        print(f"🛑 Stopping {server.name}...")
        self.running[name] = False
        
        try:
            process = self.processes[name]
            
            # Try to terminate gracefully first
            if self.is_process_running(process):
                # Kill process tree (including child processes)
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.terminate()
                    parent.terminate()
                    
                    # Wait for graceful shutdown
                    psutil.wait_procs([parent] + children, timeout=5)
                except psutil.NoSuchProcess:
                    pass
                except psutil.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    try:
                        parent = psutil.Process(process.pid)
                        children = parent.children(recursive=True)
                        for child in children:
                            child.kill()
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
            
            del self.processes[name]
            print(f"✅ {server.name} stopped successfully")
            
        except Exception as e:
            print(f"❌ Failed to stop {server.name}: {e}")

    def is_process_running(self, process: subprocess.Popen) -> bool:
        """Check if process is still running"""
        try:
            return process.poll() is None and psutil.pid_exists(process.pid)
        except:
            return False

    def _monitor_server(self, name: str):
        """Monitor server process"""
        while self.running.get(name, False):
            if name in self.processes:
                if not self.is_process_running(self.processes[name]):
                    print(f"⚠️  {self.servers[name].name} exited unexpectedly")
                    if self.running.get(name, False):
                        print(f"🔄 Restarting {self.servers[name].name} in 5 seconds...")
                        time.sleep(5)
                        if self.running.get(name, False):
                            self.start_server(name)
                    break
            time.sleep(2)

    def start_all(self):
        """Start all configured servers in order"""
        print("🚀 Starting all servers...")
        
        # Start in specific order for dependencies
        start_order = ["fastapi_gateway", "worker_deepseek", "worker_llama", "rag_server"]
        
        for name in start_order:
            if name in self.servers:
                self.start_server(name)
                time.sleep(3)  # Wait between starts
        
        print("✅ All servers startup initiated")

    def stop_all(self):
        """Stop all running servers"""
        print("🛑 Stopping all servers...")
        for name in list(self.processes.keys()):
            self.stop_server(name)
        print("✅ All servers stopped")

    def restart_server(self, name: str):
        """Restart a specific server"""
        print(f"🔄 Restarting {self.servers[name].name}...")
        self.stop_server(name)
        time.sleep(3)
        self.start_server(name)

    def status(self):
        """Show status of all servers"""
        print("\n" + "="*80)
        print("🖥️  SERVER STATUS DASHBOARD")
        print("="*80)
        
        for name, server in self.servers.items():
            if name in self.processes and self.is_process_running(self.processes[name]):
                status = "🟢 RUNNING"
                pid = f"PID: {self.processes[name].pid}"
                if server.port:
                    port_info = f"Port: {server.port}"
                else:
                    port_info = "No port"
            else:
                status = "🔴 STOPPED"
                pid = "N/A"
                port_info = "N/A"
            
            print(f"{server.name:20} | {status:12} | {pid:12} | {port_info:12}")
            print(f"{'':20} | {'Dir: ' + server.working_dir}")
            print("-" * 80)
        print()

    def interactive_menu(self):
        """Interactive command-line interface"""
        try:
            while True:
                print("\n" + "="*50)
                print("🖥️  WINDOWS SERVER MANAGER")
                print("="*50)
                print("1. 🚀 Start all servers")
                print("2. 🛑 Stop all servers") 
                print("3. 📊 Show status")
                print("4. ▶️  Start specific server")
                print("5. ⏹️  Stop specific server")
                print("6. 🔄 Restart specific server")
                print("7. 🚪 Exit")
                print("-" * 50)
                
                choice = input("Enter your choice (1-7): ").strip()
                
                if choice == '1':
                    self.start_all()
                elif choice == '2':
                    self.stop_all()
                elif choice == '3':
                    self.status()
                elif choice == '4':
                    self._show_server_list()
                    name = input("Enter server key: ").strip().lower()
                    if name in self.servers:
                        self.start_server(name)
                    else:
                        print("❌ Invalid server key")
                elif choice == '5':
                    self._show_running_servers()
                    name = input("Enter server key: ").strip().lower()
                    if name in self.servers:
                        self.stop_server(name)
                    else:
                        print("❌ Invalid server key")
                elif choice == '6':
                    self._show_server_list()
                    name = input("Enter server key: ").strip().lower()
                    if name in self.servers:
                        self.restart_server(name)
                    else:
                        print("❌ Invalid server key")
                elif choice == '7':
                    self.stop_all()
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.stop_all()

    def _show_server_list(self):
        """Show available servers"""
        print("\nAvailable servers:")
        for key, server in self.servers.items():
            print(f"  {key}: {server.name}")

    def _show_running_servers(self):
        """Show currently running servers"""
        print("\nRunning servers:")
        running_found = False
        for key, server in self.servers.items():
            if key in self.processes and self.is_process_running(self.processes[key]):
                print(f"  {key}: {server.name}")
                running_found = True
        if not running_found:
            print("  No servers currently running")

def main():
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("❌ psutil library is required. Install it with: pip install psutil")
        return

    manager = WindowsServerManager()
    
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "start":
            if len(sys.argv) > 2:
                manager.start_server(sys.argv[2])
            else:
                manager.start_all()
        elif command == "stop":
            if len(sys.argv) > 2:
                manager.stop_server(sys.argv[2])
            else:
                manager.stop_all()
        elif command == "restart":
            if len(sys.argv) > 2:
                manager.restart_server(sys.argv[2])
            else:
                manager.stop_all()
                time.sleep(3)
                manager.start_all()
        elif command == "status":
            manager.status()
        else:
            print("Usage: python windows_server_manager.py [start|stop|restart|status] [server_key]")
            print("Server keys: fastapi_gateway, worker_deepseek, worker_llama, rag_server")
    else:
        # Interactive mode
        manager.interactive_menu()

if __name__ == "__main__":
    main()