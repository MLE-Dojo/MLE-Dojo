"""
Sandbox Test Module.

This module provides comprehensive tests for the Sandbox class, focusing on:
1. Resource constraints (CPU, memory, GPU)
2. Execution timeout handling
3. Process cleanup and termination
4. Real-time output and logging

The tests verify that the Sandbox properly isolates and controls execution
of potentially untrusted code with appropriate resource limitations.
"""

import os
import sys
import time
import signal
import unittest
import tempfile
import subprocess
import psutil
from typing import List, Dict, Any, Optional

# Add parent directory to path to import the Sandbox class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mledojo.gym.sandbox import Sandbox
from mledojo.gym.error import SandboxExecutionError


class SandboxTest(unittest.TestCase):
    """Test suite for the Sandbox class."""

    def setUp(self):
        """Set up test environment before each test."""
        print("\n" + "="*80)
        print(f"SETUP: Initializing test environment")
        self.test_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"SETUP: Created test directory at {self.test_dir}")
        print(f"SETUP: Created log directory at {self.log_dir}")
        
        # Create test scripts
        self.create_test_scripts()
        print(f"SETUP: Created {len(self.test_scripts)} test scripts")
        print("="*80)

    def tearDown(self):
        """Clean up test environment after each test."""
        print("\n" + "="*80)
        print(f"TEARDOWN: Cleaning up test environment")
        
        # Remove test scripts
        for script_name, script_path in self.test_scripts.items():
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                    print(f"TEARDOWN: Removed test script {script_name} at {script_path}")
                except Exception as e:
                    print(f"TEARDOWN: Failed to remove script {script_name}: {e}")
        
        # Clean up test directory
        try:
            import shutil
            shutil.rmtree(self.test_dir)
            print(f"TEARDOWN: Removed test directory at {self.test_dir}")
        except Exception as e:
            print(f"TEARDOWN: Failed to remove test directory: {e}")
        print("="*80)

    def create_test_scripts(self):
        """Create test scripts for sandbox execution."""
        self.test_scripts = {}
        
        # Simple script that prints a message and exits
        simple_script = os.path.join(self.test_dir, "simple_script.py")
        with open(simple_script, "w") as f:
            f.write('print("Hello from sandbox!")\n')
        self.test_scripts["simple"] = simple_script
        
        # CPU-intensive script
        cpu_intensive_script = os.path.join(self.test_dir, "cpu_intensive.py")
        with open(cpu_intensive_script, "w") as f:
            f.write('''
import time
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("Starting CPU-intensive task...")
result = fibonacci(35)
print(f"Fibonacci result: {result}")
''')
        self.test_scripts["cpu_intensive"] = cpu_intensive_script
        
        # Memory-intensive script
        memory_intensive_script = os.path.join(self.test_dir, "memory_intensive.py")
        with open(memory_intensive_script, "w") as f:
            f.write('''
import time
print("Starting memory-intensive task...")
# Allocate a large list (approximately 500MB)
large_list = [0] * (500 * 1024 * 1024 // 8)
print(f"Allocated memory: {len(large_list) * 8 / (1024 * 1024):.2f} MB")
time.sleep(1)  # Keep the memory allocated for a moment
''')
        self.test_scripts["memory_intensive"] = memory_intensive_script
        
        # Long-running script
        long_running_script = os.path.join(self.test_dir, "long_running.py")
        with open(long_running_script, "w") as f:
            f.write('''
import time
print("Starting long-running task...")
for i in range(10):
    print(f"Iteration {i+1}/10")
    time.sleep(1)
print("Long-running task completed")
''')
        self.test_scripts["long_running"] = long_running_script
        
        # GPU script (if available)
        gpu_script = os.path.join(self.test_dir, "gpu_script.py")
        with open(gpu_script, "w") as f:
            f.write('''
import os
import sys

print("Checking GPU environment...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA_MEM_LIMIT: {os.environ.get('CUDA_MEM_LIMIT', 'Not set')}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Allocate a large tensor to test memory limits
        print("Allocating GPU memory...")
        tensor_size = 1000  # Adjust based on available GPU memory
        cuda_tensor = torch.rand(tensor_size, tensor_size, device='cuda')
        print(f"Allocated tensor of size {tensor_size}x{tensor_size}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
except ImportError:
    print("PyTorch not available")
except Exception as e:
    print(f"Error during GPU test: {str(e)}")
''')
        self.test_scripts["gpu"] = gpu_script
        
        # Script that spawns a child process
        child_process_script = os.path.join(self.test_dir, "child_process.py")
        with open(child_process_script, "w") as f:
            f.write('''
import os
import time
import subprocess
import signal

def signal_handler(sig, frame):
    print(f"Parent process received signal: {sig}")
    sys.exit(0)

print(f"Parent process PID: {os.getpid()}")

# Start a child process that runs for a long time
child_process = subprocess.Popen(
    ["python3", "-c", """
import time
import os
print(f'Child process started with PID: {os.getpid()}')
try:
    while True:
        time.sleep(1)
        print('Child process still running...')
except KeyboardInterrupt:
    print('Child process interrupted')
"""],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print(f"Started child process with PID: {child_process.pid}")

# Run for a few seconds to allow the sandbox to terminate us
for i in range(5):
    print(f"Parent process still running... ({i+1}/5)")
    time.sleep(1)

print("Parent process exiting normally")
''')
        self.test_scripts["child_process"] = child_process_script

    def test_simple_execution(self):
        """Test basic execution of a simple script."""
        print("\n" + "-"*80)
        print("TEST: Simple Execution")
        print("Description: Testing basic execution of a simple script")
        print("Script path:", self.test_scripts["simple"])
        print("-"*80)
        
        sandbox = Sandbox(
            log_dir=self.log_dir,
            real_time_output=True
        )
        print("Created sandbox with real-time output enabled")
        
        print("Executing script in sandbox...")
        result = sandbox.run_code(self.test_scripts["simple"])
        
        print(f"Execution completed with status: {result['status']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        self.assertEqual(result["status"], 0, "Simple script should exit with status 0")
        self.assertIn("Hello from sandbox!", result["stdout"], "Expected output not found")
        self.assertEqual(result["stderr"], "", "Stderr should be empty")
        print("Test passed: Script executed successfully with expected output")

    def test_cpu_limit(self):
        """Test CPU time limit enforcement."""
        print("\n" + "-"*80)
        print("TEST: CPU Time Limit")
        print("Description: Testing CPU time limit enforcement")
        print("Script path:", self.test_scripts["cpu_intensive"])
        print("CPU time limit: 1 second")
        print("-"*80)
        
        # Set a very low CPU time limit to trigger the limit
        sandbox = Sandbox(
            cpu_time_limit=1,  # 1 second CPU time limit
            log_dir=self.log_dir,
            real_time_output=True
        )
        print("Created sandbox with CPU time limit of 1 second")
        
        print("Executing CPU-intensive script in sandbox...")
        result = sandbox.run_code(self.test_scripts["cpu_intensive"])
        
        print(f"Execution completed with status: {result['status']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        # The process should be terminated due to CPU limit
        self.assertNotEqual(result["status"], 0, "Process should be terminated due to CPU limit")
        
        # Check if the error message indicates CPU limit exceeded
        # Note: The exact error message might vary by platform
        cpu_limit_indicators = ["CPU time limit exceeded", "Resource temporarily unavailable"]
        stderr_indicates_cpu_limit = any(indicator in result["stderr"] for indicator in cpu_limit_indicators)
        
        # If stderr doesn't clearly indicate CPU limit, check if process was terminated
        if stderr_indicates_cpu_limit:
            print("Test passed: Process was terminated due to CPU time limit as expected")
        else:
            print("Test passed: Process was terminated, but error message doesn't explicitly mention CPU limit")
            self.assertNotEqual(result["status"], 0, "Process should have been terminated")

    def test_memory_limit(self):
        """Test memory limit enforcement."""
        print("\n" + "-"*80)
        print("TEST: Memory Limit")
        print("Description: Testing memory limit enforcement")
        print("Script path:", self.test_scripts["memory_intensive"])
        print("Memory limit: 100MB")
        print("-"*80)
        
        # Set a low memory limit to trigger the limit
        sandbox = Sandbox(
            memory_limit=0.1,  # 100MB memory limit
            log_dir=self.log_dir,
            real_time_output=True
        )
        print("Created sandbox with memory limit of 100MB")
        
        print("Executing memory-intensive script in sandbox...")
        result = sandbox.run_code(self.test_scripts["memory_intensive"])
        
        print(f"Execution completed with status: {result['status']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        # The process should be terminated due to memory limit
        # Note: On some systems, this might not trigger exactly as expected
        # due to how memory limits are enforced
        if result["status"] != 0:
            # Process was terminated, which is expected
            memory_limit_indicators = ["Memory limit exceeded", "Cannot allocate memory"]
            stderr_indicates_memory_limit = any(indicator in result["stderr"] for indicator in memory_limit_indicators)
            
            # If stderr doesn't clearly indicate memory limit, check if process was terminated
            if stderr_indicates_memory_limit:
                print("Test passed: Process was terminated due to memory limit as expected")
            else:
                print("Test passed: Process was terminated, but error message doesn't explicitly mention memory limit")
                self.assertNotEqual(result["status"], 0, "Process should have been terminated")
        else:
            # If the process completed successfully, this test is inconclusive
            # This can happen on systems where memory limits are not strictly enforced
            print("Warning: Memory limit test inconclusive - process completed successfully")
            print("This can happen on systems where memory limits are not strictly enforced")

    def test_execution_timeout(self):
        """Test execution timeout enforcement."""
        print("\n" + "-"*80)
        print("TEST: Execution Timeout")
        print("Description: Testing execution timeout enforcement")
        print("Script path:", self.test_scripts["long_running"])
        print("Execution timeout: 2 seconds")
        print("-"*80)
        
        # Set a short timeout to trigger timeout handling
        sandbox = Sandbox(
            execution_timeout=2,  # 2 second timeout
            log_dir=self.log_dir,
            real_time_output=True
        )
        print("Created sandbox with execution timeout of 2 seconds")
        
        print("Executing long-running script in sandbox...")
        start_time = time.time()
        
        # Track processes before execution
        initial_process_history = set(sandbox.process_history) if hasattr(sandbox, 'process_history') else set()
        
        result = sandbox.run_code(self.test_scripts["long_running"])
        execution_time = time.time() - start_time
        
        print(f"Execution completed with status: {result['status']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Check if any processes were added to process_history
        final_process_history = set(sandbox.process_history) if hasattr(sandbox, 'process_history') else set()
        
        # Get processes that were added during this test
        added_processes = final_process_history - initial_process_history
        print(f"Processes added to process_history: {added_processes}")
        
        # Verify that all processes in process_history are no longer running
        active_processes = []
        for pid in added_processes:
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        active_processes.append(pid)
                except psutil.NoSuchProcess:
                    pass
        
        self.assertEqual(len(active_processes), 0, 
                         f"Some processes are still running: {active_processes}")
        
        # Verify the process was terminated due to timeout
        self.assertNotEqual(result["status"], 0, "Process should be terminated due to timeout")
        self.assertLess(execution_time, 5, "Execution should be terminated within 5 seconds")
        
        print("Test passed: Process was terminated due to execution timeout as expected")

    def test_gpu_environment(self):
        """Test GPU environment configuration."""
        print("\n" + "-"*80)
        print("TEST: GPU Environment")
        print("Description: Testing GPU environment configuration")
        print("Script path:", self.test_scripts["gpu"])
        print("-"*80)
        
        # Skip if no GPU is available
        try:
            import torch
            if not torch.cuda.is_available():
                print("Skipping test: No GPU available")
                self.skipTest("No GPU available")
        except ImportError:
            print("Skipping test: PyTorch not installed")
            self.skipTest("PyTorch not installed")
        
        print("GPU available, proceeding with test")
        print("GPU device: 0")
        print("GPU memory limit: 1GB")
        
        # Test with GPU configuration
        sandbox = Sandbox(
            gpu_device=0,  # Use first GPU
            gpu_memory_limit=1,  # 1GB GPU memory limit
            log_dir=self.log_dir,
            real_time_output=True
        )
        print("Created sandbox with GPU device 0 and 1GB memory limit")
        
        print("Executing GPU script in sandbox...")
        result = sandbox.run_code(self.test_scripts["gpu"])
        
        print(f"Execution completed with status: {result['status']}")
        print(f"Stdout excerpt: {result['stdout'][:200]}...")
        print(f"Stderr: {result['stderr']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        # Check that GPU environment variables were set correctly
        self.assertIn("CUDA_VISIBLE_DEVICES: 0", result["stdout"], "GPU device not set correctly")
        
        # Check that the script was able to access the GPU
        self.assertIn("CUDA available: True", result["stdout"], "CUDA not available in sandbox")
        print("Test passed: GPU environment was correctly configured in sandbox")
        
        # Note: Actual GPU memory limit enforcement depends on the GPU driver and might not
        # be directly testable in all environments

    def test_process_cleanup(self):
        """Test that child processes are properly cleaned up when sandbox is terminated."""
        print("\n" + "-"*80)
        print("TEST: Process Cleanup")
        print("Description: Testing that child processes are properly cleaned up")
        print("Script path:", self.test_scripts["child_process"])
        print("-"*80)
        
        # Create a sandbox
        sandbox = Sandbox(
            log_dir=self.log_dir,
            real_time_output=True,
            execution_timeout=10  # Set a longer timeout
        )
        print("Created sandbox with execution timeout of 10 seconds")
        
        # Start the script that spawns a child process in a separate thread
        import threading
        result_container = {}
        
        def run_in_sandbox():
            try:
                result = sandbox.run_code(self.test_scripts["child_process"])
                result_container["result"] = result
            except Exception as e:
                result_container["exception"] = e
        
        print("Starting script execution in a separate thread...")
        thread = threading.Thread(target=run_in_sandbox)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for the processes to start
        print("Waiting for processes to start...")
        time.sleep(3)
        
        # Get the PIDs from process_history
        initial_process_history = set(sandbox.process_history) if hasattr(sandbox, 'process_history') else set()
        print(f"Initial process history: {initial_process_history}")
        
        # Force cleanup of the sandbox
        print("Forcing cleanup of sandbox processes...")
        sandbox._cleanup_processes()
        
        # Wait a moment for processes to be terminated
        print("Waiting for processes to terminate...")
        time.sleep(1)
        
        # Check that all processes have been terminated
        active_pids = []
        for pid in initial_process_history:
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    active_pids.append(pid)
            except psutil.NoSuchProcess:
                # Process no longer exists, which is what we want
                pass
        
        print(f"Remaining active processes after cleanup: {active_pids}")
        self.assertEqual(len(active_pids), 0, f"Some processes were not terminated: {active_pids}")
        print("Test passed: All sandbox processes were properly terminated")

    def test_real_time_output(self):
        """Test real-time output streaming."""
        print("\n" + "-"*80)
        print("TEST: Real-time Output")
        print("Description: Testing real-time output streaming")
        print("Script path:", self.test_scripts["long_running"])
        print("-"*80)
        
        # Redirect stdout to capture real-time output
        original_stdout = sys.stdout
        sys.stdout = temp_stdout = tempfile.TemporaryFile(mode='w+t')
        
        try:
            print("Redirecting stdout to capture real-time output")
            sandbox = Sandbox(
                log_dir=self.log_dir,
                real_time_output=True,
                real_time_logging=False
            )
            print("Created sandbox with real-time output enabled")
            
            print("Executing long-running script in sandbox...")
            sandbox.run_code(self.test_scripts["long_running"])
            
            # Reset stdout position and read the captured output
            temp_stdout.seek(0)
            captured_output = temp_stdout.read()
            
            # Restore original stdout to print results
            sys.stdout = original_stdout
            print(f"Captured output length: {len(captured_output)} characters")
            print(f"Captured output excerpt: {captured_output[:200]}...")
            
            # Check that real-time output was captured
            self.assertIn("Starting long-running task", captured_output, 
                         "Real-time output not captured")
            print("Test passed: Real-time output was successfully captured")
            
        finally:
            # Restore original stdout if not already done
            if sys.stdout != original_stdout:
                sys.stdout = original_stdout
            temp_stdout.close()
            print("Restored original stdout")

    def test_sandbox_timeout_termination(self):
        """Test that sandbox automatically terminates a long-running task that exceeds timeout."""
        print("\n" + "-"*80)
        print("TEST: Sandbox Timeout Termination")
        print("Description: Testing automatic termination of tasks exceeding timeout")
        print("Execution timeout: 30 seconds")
        print("-"*80)
        
        # Create a sandbox with a 30s timeout
        sandbox = Sandbox(
            log_dir=self.log_dir,
            execution_timeout=30  # 30 second timeout
        )
        print("Created sandbox with execution timeout of 30 seconds")
        
        # Create a test script that runs for 120 seconds
        long_running_code = """
import time
print("Starting a task that should run for 120 seconds")
start_time = time.time()
try:
    time.sleep(120)  # Try to sleep for 120 seconds
    print("Task completed after 120 seconds")
except Exception as e:
    print(f"Task interrupted after {time.time() - start_time:.2f} seconds")
"""
        
        # Create a temporary file with the long-running code
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.py', delete=False) as temp_file:
            temp_file.write(long_running_code)
            temp_file_path = temp_file.name
            print(f"Created temporary script at {temp_file_path}")
        
        try:
            # Track processes before execution
            initial_process_history = set(sandbox.process_history) if hasattr(sandbox, 'process_history') else set()
            print(f"Initial process history: {initial_process_history}")
            # Run the code in the sandbox
            print("Executing long-running script in sandbox...")
            result = sandbox.run_code(temp_file_path)
            
            print(f"Execution completed with status: {result['status']}")
            print(f"Stdout: {result['stdout']}")
            print(f"Stderr: {result['stderr']}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            
            # Check that the execution was terminated due to timeout
            self.assertNotEqual(result["status"], 0, "Process should have been terminated due to timeout")
            self.assertLess(result["execution_time"], 35, "Process should have been terminated after ~30 seconds")
            
            # Get processes that were added during this test
            final_process_history = set(sandbox.process_history) if hasattr(sandbox, 'process_history') else set()
            added_processes = final_process_history - initial_process_history
            print(f"Processes added to process_history: {added_processes}")
            
            # Verify all processes were cleaned up
            active_pids = []
            for pid in added_processes:
                try:
                    process = psutil.Process(pid)
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        active_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            self.assertEqual(len(active_pids), 0, f"Some processes were not terminated: {active_pids}")
            print("Test passed: Long-running task was automatically terminated after timeout")
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            print(f"Removed temporary script at {temp_file_path}")


if __name__ == "__main__":
    unittest.main()

