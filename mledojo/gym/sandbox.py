"""
MLE-Dojo Sandbox Module.

This module provides a secure sandbox environment for executing untrusted code with
resource limitations and monitoring. The Sandbox class enables controlled execution
of Python code with configurable CPU, memory, and GPU resource constraints.

Typical usage:
    # Create a sandbox with resource limits
    sandbox = Sandbox(
        cpu_time_limit=30,
        memory_limit=2,
        execution_timeout=60
    )
    
    # Execute code within the sandbox
    result = sandbox.run_code("user_submission.py")
    
    # Process results
    if result["status"] == 0:
        print("Execution successful")
    else:
        print(f"Execution failed: {result['stderr']}")
"""

import subprocess
import os
import signal
import logging
import tempfile
import time
import resource
import threading
import sys
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Tuple
import signal as signal_module  # Renamed to avoid conflict
from contextlib import contextmanager

from mledojo.gym.error import (
    SandboxError,
    SandboxInitializationError, 
    SandboxExecutionError,
    SandboxResourceError,
    SandboxCleanupError
)

# Type definitions for better code readability
ProcessType = TypeVar('ProcessType', bound=subprocess.Popen)
ResourceLimits = Dict[int, Tuple[int, int]]


class Sandbox:
    """Secure execution environment with resource constraints.
    
    The Sandbox provides a controlled environment for executing potentially untrusted
    code with configurable resource limitations. It supports CPU time limits, memory
    limits, execution timeouts, and GPU resource constraints. All execution is isolated
    and monitored, with comprehensive logging capabilities.
    
    Attributes:
        gpu_device (Optional[int]): GPU device ID to use, None if no GPU is needed.
        gpu_memory_limit (Optional[int]): GPU memory limit in bytes.
        cpu_time_limit (Optional[int]): CPU time limit in seconds.
        memory_limit (Optional[int]): Memory limit in bytes.
        execution_timeout (Optional[int]): Maximum execution time in seconds.
        log_dir (str): Directory for storing execution logs.
        real_time_output (bool): Whether to show output in real-time.
        real_time_logging (bool): Whether to log output in real-time.
        logger (logging.Logger): Logger for sandbox operations.
        active_processes (Dict[int, ProcessType]): Currently running processes mapped by PID.
        process_history (List[int]): History of all process PIDs that have been created.
    """

    def __init__(
        self, 
        gpu_device: Optional[int] = None, 
        gpu_memory_limit: Optional[int] = None,
        cpu_time_limit: Optional[int] = None, 
        memory_limit: Optional[int] = None,
        execution_timeout: Optional[int] = None, 
        log_dir: Optional[str] = None,
        real_time_output: bool = False, 
        real_time_logging: bool = False
    ):
        """Initialize a sandbox environment for code execution with resource constraints.
        
        Args:
            gpu_device: GPU device ID to use (None means no GPU).
            gpu_memory_limit: GPU memory limit in GB.
            cpu_time_limit: CPU time limit in seconds.
            memory_limit: Memory limit in GB.
            execution_timeout: Timeout for code execution in seconds.
            log_dir: Directory to store execution logs (None uses system temp dir).
            real_time_output: Whether to display subprocess output in real-time.
            real_time_logging: Whether to log subprocess output in real-time.
            
        Raises:
            SandboxInitializationError: If initialization fails due to invalid resource limits,
                log directory creation failure, or GPU unavailability.
        """
        try:
            # Convert GB to bytes for internal storage
            self.gpu_device = gpu_device
            self.gpu_memory_limit = self._gb_to_bytes(gpu_memory_limit) if gpu_memory_limit else None
            self.cpu_time_limit = cpu_time_limit
            self.memory_limit = self._gb_to_bytes(memory_limit) if memory_limit else None
            self.execution_timeout = execution_timeout
            self.real_time_output = real_time_output
            self.real_time_logging = real_time_logging
            
            # Setup logging infrastructure
            self.logger = self._setup_logger()
            self.log_dir = log_dir or tempfile.gettempdir()
            self._ensure_log_directory()
            
            # Keep track of active processes for cleanup, mapped by PID for one-to-one identification
            self.active_processes: Dict[int, ProcessType] = {}
            
            # Keep track of all process PIDs that have been created
            self.process_history: List[int] = []
            
            # Register cleanup handler for when the parent process exits
            self._register_cleanup_handlers()
            
            # Add signal handlers for proper termination
            self._register_signal_handlers()
            
        except Exception as e:
            raise SandboxInitializationError(
                message=f"Failed to initialize sandbox: {str(e)}",
                details=(
                    f"GPU device: {gpu_device}, "
                    f"Memory limit: {memory_limit}GB, "
                    f"CPU time limit: {cpu_time_limit}s"
                )
            )
    
    def _gb_to_bytes(self, value: int) -> int:
        """Convert GB value to bytes.
        
        Args:
            value: Value in gigabytes.
            
        Returns:
            Equivalent value in bytes.
        """
        return value * 1024 * 1024 * 1024
    
    def _bytes_to_gb(self, value: int) -> float:
        """Convert bytes value to GB.
        
        Args:
            value: Value in bytes.
            
        Returns:
            Equivalent value in gigabytes.
        """
        return value / 1024 / 1024 / 1024
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger for sandbox operations.
        
        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger("mledojo.sandbox")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers if logger already exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)
            # Prevent messages from propagating to ancestor loggers (like root)
            # This stops duplicate console output if root/parent loggers also have StreamHandlers
            logger.propagate = False
            
        # If handlers already exist, assume propagation is handled correctly elsewhere or is desired.
        # We only set propagate=False when we *add* the handler here.
            
        return logger
    
    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists.
        
        Raises:
            SandboxInitializationError: If log directory creation fails.
        """
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise SandboxInitializationError(
                message=f"Failed to create log directory: {str(e)}",
                details=f"Log directory path: {self.log_dir}"
            )
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for process termination.
        
        Ensures that processes are properly terminated when the sandbox or
        parent process exits.
        """
        import atexit
        atexit.register(self._cleanup_processes)
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers to ensure cleanup on termination."""
        # Store original signal handlers to restore them later if needed
        self._original_sigterm = signal_module.getsignal(signal_module.SIGTERM)
        self._original_sigint = signal_module.getsignal(signal_module.SIGINT)
        self._original_sighup = signal_module.getsignal(signal_module.SIGHUP)
        
        # Set new signal handlers
        signal_module.signal(signal_module.SIGTERM, self._signal_handler)
        signal_module.signal(signal_module.SIGINT, self._signal_handler)
        signal_module.signal(signal_module.SIGHUP, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals by cleaning up processes first."""
        self.logger.info(f"Received signal {signum}, cleaning up processes before exit")
        
        # Force termination even if there are errors
        try:
            self._cleanup_processes(force=True)
        except Exception as e:
            self.logger.error(f"Error during forced cleanup: {str(e)}")
            # Continue with exit even if cleanup fails
        
        # Call original handler if it was a callable
        if signum == signal_module.SIGTERM:
            original_handler = self._original_sigterm
        elif signum == signal_module.SIGINT:
            original_handler = self._original_sigint
        elif signum == signal_module.SIGHUP:
            original_handler = self._original_sighup
        else:
            original_handler = None
        
        if callable(original_handler):
            original_handler(signum, frame)
        else:
            # Default behavior - exit with signal number as status
            sys.exit(128 + signum)
    
    def _cleanup_processes(self, force=False) -> None:
        """Clean up any active processes when the sandbox is destroyed or parent process exits.
        
        Args:
            force: If True, use more aggressive termination strategies
            
        Raises:
            SandboxCleanupError: If process termination fails and force is False.
        """
        success = True
        errors = []
        
        for pid, process in list(self.active_processes.items()):
            if process and process.poll() is None:
                try:
                    # Kill the entire process group to ensure all child processes are terminated
                    self.logger.info(f"Cleaning up process {pid}")
                    
                    # First try SIGTERM for graceful shutdown
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    
                    # Give the process a short time to terminate gracefully
                    for _ in range(3):  # Try for up to 0.3 seconds
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    
                    # If still running, use SIGKILL for forced termination
                    if process.poll() is None:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        
                        # Verify process is actually terminated
                        for _ in range(5):  # Try for up to 0.5 seconds
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                        
                        # Final verification and extra measure if still running
                        if process.poll() is None:
                            # Try direct process kill as last resort
                            if force:
                                try:
                                    process.kill()
                                    # One last check
                                    time.sleep(0.1)
                                except Exception as e:
                                    self.logger.error(f"Final kill attempt failed for PID {pid}: {str(e)}")
                            
                            if process.poll() is None:
                                self.logger.error(f"Failed to terminate process {pid} with SIGKILL")
                                success = False
                                errors.append(f"PID {pid} refuses to terminate")
                except (ProcessLookupError, PermissionError, OSError) as e:
                    error_msg = f"Failed to terminate process {pid}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    success = False
                finally:
                    # Always remove from active processes dictionary regardless of success
                    # This prevents retrying dead processes
                    if pid in self.active_processes:
                        del self.active_processes[pid]
        
        # If force=False and there were errors, raise exception
        if not success and not force:
            raise SandboxCleanupError(
                message="Failed to clean up all processes",
                details="; ".join(errors)
            )
    
    def _setup_resource_limits(self) -> ResourceLimits:
        """Set up resource limits for the child process.
        
        Configures memory and CPU time limits for the subprocess.
        
        Returns:
            Dictionary mapping resource types to (soft, hard) limit tuples.
        """
        limits: ResourceLimits = {}
        
        if self.memory_limit:
            # Convert bytes to kilobytes for setrlimit
            kb_limit = int(self.memory_limit)
            limits[resource.RLIMIT_AS] = (kb_limit, kb_limit)
            
        if self.cpu_time_limit:
            limits[resource.RLIMIT_CPU] = (self.cpu_time_limit, self.cpu_time_limit)
            
        return limits
        
    def _preexec_fn(self) -> None:
        """Function to be called in the child process before execution.
        
        Sets up resource limits and process group for the child process.
        
        Raises:
            SandboxResourceError: If resource limit configuration fails.
        """
        try:
            # Set resource limits in the child process
            limits = self._setup_resource_limits()
            for resource_type, (soft, hard) in limits.items():
                resource.setrlimit(resource_type, (soft, hard))
            
            # Set process group for easier cleanup
            os.setpgrp()
        except Exception as e:
            # This exception will be caught by the subprocess module
            sys.stderr.write(f"Failed to set resource limits: {str(e)}\n")
            sys.exit(1)
    
    def _stream_output(
        self, 
        pipe: Any, 
        is_stderr: bool = False, 
        logger: Optional[logging.Logger] = None
    ) -> str:
        """Stream output from subprocess pipe to console and internal buffer.
        
        Args:
            pipe: Subprocess pipe (stdout or stderr).
            is_stderr: Whether the pipe is stderr (vs stdout).
            logger: Logger to use for real-time logging.
            
        Returns:
            Captured output as a string.
        """
        buffer = []
        for line in iter(pipe.readline, ''):
            if not line:
                break
            buffer.append(line)
            
            # Output to console if real_time_output is enabled
            if self.real_time_output:
                if is_stderr:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            
            # Log in real-time if enabled
            if self.real_time_logging and logger:
                if is_stderr:
                    logger.error(f"STDERR (real-time): {line.strip()}")
                else:
                    logger.info(f"STDOUT (real-time): {line.strip()}")
                
        return ''.join(buffer)
    
    def _prepare_execution_environment(
        self, 
        code_file: str
    ) -> Tuple[List[str], Dict[str, str], str]:
        """Prepare the execution environment for running code.
        
        Sets up command, environment variables, and log file for execution.
        
        Args:
            code_file: Path to code file to execute.
            
        Returns:
            Tuple containing (cmd, env, log_filename).
        """
        # Create a unique log file for this execution
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(self.log_dir, f"sandbox_run_{timestamp}.log")
        
        # Prepare command - use absolute path to avoid numpy import error
        code_file_abs = os.path.abspath(code_file)
        cmd = ["python3", code_file_abs]
        
        # Prepare environment variables
        env = os.environ.copy()
        
        # Configure GPU environment if needed
        if self.gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)
            if self.gpu_memory_limit:
                env["CUDA_MEM_LIMIT"] = str(self.gpu_memory_limit)
        
        return cmd, env, log_filename
    
    def _setup_execution_logging(self, log_filename: str) -> logging.FileHandler:
        """Set up logging for code execution.
        
        Args:
            log_filename: Path to log file.
            
        Returns:
            Log file handler.
        """
        log_file_handler = logging.FileHandler(log_filename)
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_file_handler)
        return log_file_handler
    
    def _log_execution_start(self, code_file: str, cmd: List[str]) -> None:
        """Log execution start information.
        
        Args:
            code_file: Path to code being executed.
            cmd: Command being run.
        """
        self.logger.info(f"Starting execution of {code_file}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(
            f"Resource limits: CPU={self.cpu_time_limit}s, "
            f"GPU={self.gpu_device}, "
            f"GPU MEM={self._bytes_to_gb(self.gpu_memory_limit):.2f}GB, " if self.gpu_memory_limit else "GPU MEM=None, "
            f"Execution timeout={self.execution_timeout}s, "
            # f"MEM={self._bytes_to_gb(self.memory_limit):.2f}GB" if self.memory_limit 
            # else f"Resource limits: CPU={self.cpu_time_limit}s, MEM=None"
        )
    
    def _run_with_streaming(
        self, 
        cmd: List[str], 
        env: Dict[str, str]
    ) -> Tuple[ProcessType, str, str]:
        """Run command with real-time output streaming.
        
        Args:
            cmd: Command to execute.
            env: Environment variables.
            
        Returns:
            Tuple of (process, stdout_data, stderr_data).
            
        Raises:
            SandboxExecutionError: If execution times out.
        """
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env, 
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=self._preexec_fn,
            cwd=os.path.dirname(os.path.abspath(cmd[1]))  # Set working directory to script location
        )
        
        # Add to active processes dictionary with PID as key
        self.active_processes[process.pid] = process
        # Add to process history
        self.process_history.append(process.pid)
        
        # Use threads to stream output in real-time
        self._stdout_data = ""
        self._stderr_data = ""
        
        stdout_thread = threading.Thread(
            target=lambda: setattr(self, '_stdout_data', 
                                  self._stream_output(process.stdout, logger=self.logger))
        )
        stderr_thread = threading.Thread(
            target=lambda: setattr(self, '_stderr_data', 
                                  self._stream_output(process.stderr, is_stderr=True, logger=self.logger))
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Create a timer to kill the process if it exceeds the timeout
        timer = None
        if self.execution_timeout:
            def kill_on_timeout():
                if process.poll() is None:  # If process is still running
                    self.logger.error(f"Process execution timed out after {self.execution_timeout}s")
                    self._terminate_process(process)
            
            timer = threading.Timer(self.execution_timeout, kill_on_timeout)
            timer.daemon = True
            timer.start()
        
        try:
            # Wait for process to complete
            process.wait()
            # Cancel the timer if process completes before timeout
            if timer:
                timer.cancel()
            # Wait for output threads to complete
            stdout_thread.join()
            stderr_thread.join()
            stdout_data = self._stdout_data
            stderr_data = self._stderr_data
        except Exception as e:
            # Cancel the timer if there's an exception
            if timer:
                timer.cancel()
            # Make sure process is terminated
            self._terminate_process(process)
            raise SandboxExecutionError(
                message=f"Process execution failed: {str(e)}",
                details=f"Command: {' '.join(cmd)}"
            )
            
        return process, stdout_data, stderr_data
    
    def _run_without_streaming(
        self, 
        cmd: List[str], 
        env: Dict[str, str]
    ) -> Tuple[ProcessType, str, str]:
        """Run command without real-time output streaming.
        
        Args:
            cmd: Command to execute.
            env: Environment variables.
            
        Returns:
            Tuple of (process, stdout_data, stderr_data).
        """
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env, 
            text=True,
            preexec_fn=self._preexec_fn,
            cwd=os.path.dirname(os.path.abspath(cmd[1]))  # Set working directory to script location
        )
        
        # Add to active processes dictionary with PID as key
        self.active_processes[process.pid] = process
        # Add to process history
        self.process_history.append(process.pid)
        
        # Create a timer to kill the process if it exceeds the timeout
        timer = None
        if self.execution_timeout:
            def kill_on_timeout():
                if process.poll() is None:  # If process is still running
                    self.logger.error(f"Process execution timed out after {self.execution_timeout}s")
                    self._terminate_process(process)
            
            timer = threading.Timer(self.execution_timeout, kill_on_timeout)
            timer.daemon = True
            timer.start()
        
        try:
            # Use communicate with timeout for safer execution
            stdout_data, stderr_data = process.communicate(timeout=self.execution_timeout)
            # Cancel the timer if process completes before timeout
            if timer:
                timer.cancel()
            return process, stdout_data, stderr_data
        except subprocess.TimeoutExpired:
            # Cancel the timer as we're handling the timeout here
            if timer:
                timer.cancel()
            # Properly terminate the process if timeout occurs
            self._terminate_process(process)
            raise SandboxExecutionError(
                message=f"Process execution timed out after {self.execution_timeout}s",
                details=f"Command: {' '.join(cmd)}"
            )
        except Exception as e:
            # Cancel the timer if there's an exception
            if timer:
                timer.cancel()
            # Make sure process is terminated
            self._terminate_process(process)
            raise SandboxExecutionError(
                message=f"Process execution failed: {str(e)}",
                details=f"Command: {' '.join(cmd)}"
            )
    
    def _terminate_process(self, process: ProcessType) -> None:
        """Safely terminate a process.
        
        Args:
            process: Process to terminate.
            
        Raises:
            SandboxCleanupError: If process termination fails.
        """
        if process and process.poll() is None:
            try:
                pid = process.pid
                # First try SIGTERM for graceful shutdown
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                
                # Give the process a short time to terminate gracefully
                for _ in range(3):  # Try for up to 0.3 seconds
                    if process.poll() is not None:
                        self.logger.info(f"Process {pid} terminated successfully with SIGTERM")
                        break
                    time.sleep(0.1)
                
                # If still running, use SIGKILL for forced termination
                if process.poll() is None:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    
                    # Verify process is actually terminated
                    for _ in range(5):  # Try for up to 0.5 seconds
                        if process.poll() is not None:
                            self.logger.info(f"Process {pid} terminated successfully with SIGKILL")
                            break
                        time.sleep(0.1)
                    
                    # Try direct process kill as final attempt
                    if process.poll() is None:
                        process.kill()
                        time.sleep(0.1)
                        
                        if process.poll() is not None:
                            self.logger.info(f"Process {pid} terminated successfully with direct kill")
                        else:
                            # If still running after SIGKILL, log an error
                            self.logger.error(f"Failed to terminate process {pid} with SIGKILL")
                            raise SandboxCleanupError(
                                message=f"Failed to terminate process {pid}",
                                details="Process survived SIGTERM, SIGKILL, and direct kill"
                            )
                
                # Remove from active processes dictionary
                if pid in self.active_processes:
                    del self.active_processes[pid]
            except (ProcessLookupError, PermissionError, OSError) as e:
                self.logger.error(f"Failed to terminate process {process.pid}: {str(e)}")
                raise SandboxCleanupError(
                    message=f"Failed to terminate process {process.pid}",
                    details=str(e)
                )
    
    def run_code(self, code_file: str) -> Dict[str, Any]:
        """Execute code file in a sandboxed subprocess with resource constraints.
        
        Args:
            code_file: Path to the Python file to execute.
            
        Returns:
            Dictionary containing execution results including:
            - status: Return code (0 for success)
            - stdout: Standard output
            - stderr: Standard error
            - execution_time: Time taken in seconds
            - log_file: Path to the log file
            
        Raises:
            SandboxExecutionError: If code execution fails due to process creation failure,
                resource limits exceeded, timeout, or termination failure.
            SandboxResourceError: If memory, CPU time, or GPU memory limits are exceeded.
        """
        # Prepare execution environment
        cmd, env, log_filename = self._prepare_execution_environment(code_file)
        
        # Setup logging
        log_file_handler = self._setup_execution_logging(log_filename)
        
        # Log execution start
        self._log_execution_start(code_file, cmd)
        
        start_time = time.time()
        process = None
        
        try:
            # Run the code with appropriate strategy
            if self.real_time_output or self.real_time_logging:
                process, stdout_data, stderr_data = self._run_with_streaming(cmd, env)
            else:
                process, stdout_data, stderr_data = self._run_without_streaming(cmd, env)
            
            # Log the output (if not already logged in real-time)
            if not self.real_time_logging:
                if stdout_data:
                    self.logger.info(f"STDOUT:\n{stdout_data}")
                if stderr_data:
                    self.logger.error(f"STDERR:\n{stderr_data}")
                
            execution_time = time.time() - start_time
            self.logger.info(f"Execution completed in {execution_time:.2f}s with return code {process.returncode}")
            
            # Remove from active processes dictionary since it completed
            if process.pid in self.active_processes:
                del self.active_processes[process.pid]
            
            return {
                "status": process.returncode,
                "stdout": stdout_data,
                "stderr": stderr_data,
                "execution_time": execution_time,
                "log_file": log_filename
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.logger.error(f"Execution timed out after {execution_time:.2f}s")
            
            # Ensure process is properly terminated
            if process:
                self._terminate_process(process)
                    
            return {
                "status": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {execution_time:.2f}s",
                "execution_time": execution_time,
                "log_file": log_filename
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Cleanup if process exists
            if process:
                self._terminate_process(process)
                    
            return {
                "status": -1,
                "stdout": "",
                "stderr": error_msg,
                "execution_time": execution_time,
                "log_file": log_filename
            }
            
        finally:
            # Clean up logging handler
            self.logger.removeHandler(log_file_handler)
            log_file_handler.close()
            
    def __del__(self):
        """Clean up resources when Sandbox is garbage collected."""
        try:
            self._cleanup_processes()
        except Exception as e:
            # Just log the error, don't raise during garbage collection
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during sandbox cleanup: {str(e)}")
    
    def get_process_info(self) -> Dict[str, Any]:
        """Return information about active processes and process history.
        
        Returns:
            Dictionary containing active_processes and process_history.
        """
        return {
            "active_processes": list(self.active_processes.keys()),
            "process_history": self.process_history
        }
    
    @contextmanager
    def safe_execution(self):
        """Context manager for safe execution with guaranteed cleanup.
        
        Usage:
            with sandbox.safe_execution():
                # Run code that might be interrupted
        """
        try:
            yield
        finally:
            self._cleanup_processes()