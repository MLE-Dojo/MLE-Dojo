#!/usr/bin/env python3
import subprocess
import time
import threading
import argparse
from queue import Queue
from datetime import datetime, timedelta
import os
import json
import logging
from pathlib import Path
import signal  # Add signal module
import sys     # Add sys module for clean exit

# Resolve the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()

# Set up logging
def setup_logging(log_dir):
    """Configures the logging system."""
    log_path = Path(log_dir).resolve() / "manager.log"
    log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

    # Create logger
    logger = logging.getLogger("DockerTaskManager")
    logger.setLevel(logging.INFO) # Set the minimum level to log

    # Prevent adding multiple handlers if called again (though unlikely here)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Docker Task Manager for GPU workloads')
    parser.add_argument('--gpu-indices', type=int, nargs='+', default=[0, 1],
                        help='List of GPU indices available')
    parser.add_argument('--max-tasks-per-gpu', type=int, default=2,
                        help='Maximum tasks per GPU')
    parser.add_argument('--docker-image', type=str, default="mle-dojo",
                        help='Docker image to use')
    parser.add_argument('--competitions', type=str, nargs='+', default=["spaceship-titanic"],
                        help='List of competition names to run')
    parser.add_argument('--competitions-file', type=str,
                        help='Path to a text file with competition names (one per line)')
    parser.add_argument('--task-runtime-seconds', type=int, default=3600,
                        help='Maximum allowed runtime (in seconds) for each docker container')
    parser.add_argument('--log-dir', type=str, default="./experiments/logs",
                        help='Directory for storing log files (relative or absolute)')
    parser.add_argument('--kaggle-data', type=str, default="./data/prepare",
                        help='Directory for storing kaggle data (relative or absolute)')
    parser.add_argument('--output-dir', type=str, default="./experiments/results",
                        help='Directory for storing output files (relative or absolute)')
    return parser.parse_args()

# Use resolved absolute paths for volumes
DOCKER_RUN_TEMPLATE = " ".join((
    "docker run --detach",
    "--network host",
    "--gpus '\"device={gpu_index}\"'",
    "--shm-size=64g",
    "-v {main_py_path}:/home/main.py",
    "-v {output_dir}:/home/output",
    "-v {config_yaml_path}:/home/config.yaml",
    "-v {agent_dir_path}:/home/mledojo/agent/",
    "-v {kaggle_data}:/home/data:ro",
    "-v {env_path}:/home/.env",
    "-e COMPETITION_NAME=\"{competition_name}\"",
    "-it {docker_image}",
))

# Define a Task class to hold task details
class Task:
    def __init__(self, task_id, competition_name, gpu_index=None):
        self.task_id = task_id
        self.gpu_index = gpu_index
        self.competition_name = competition_name
        self.start_time = None
        self.container_id = None
        self.log_file = None
        self.log_thread = None

    def to_dict(self):
        """Convert task to dictionary for logging"""
        return {
            "task_id": self.task_id,
            "gpu_index": self.gpu_index,
            "competition_name": self.competition_name,
            "container_id": self.container_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "runtime": str(datetime.now() - self.start_time) if self.start_time else None
        }

# The main manager class
class DockerTaskManager:
    def __init__(self, gpu_indices, max_tasks_per_gpu, task_runtime, competitions, log_dir, kaggle_data, output_dir, docker_image):
        self.gpu_indices = gpu_indices
        self.max_tasks_per_gpu = max_tasks_per_gpu
        # Resolve and store paths as absolute Path objects early
        self.log_dir = Path(log_dir).resolve()
        self.logger = setup_logging(self.log_dir) # Setup and get logger
        self.task_runtime = task_runtime
        self.competitions = competitions
        self.docker_image = docker_image # Store docker image name

        # Resolve and store paths as absolute Path objects
        self.container_logs_dir = self.log_dir / "logs"
        self.kaggle_data = Path(kaggle_data).resolve()
        self.output_dir = Path(output_dir).resolve()

        # Resolve paths relative to the script directory for mounting
        self.main_py_path = SCRIPT_DIR / "main.py"
        self.config_yaml_path = SCRIPT_DIR / "config.yaml"
        self.agent_dir_path = SCRIPT_DIR / "mledojo" / "agent"
        self.env_path = SCRIPT_DIR / ".env"
        # Create log directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.container_logs_dir.mkdir(exist_ok=True)
        
        # Task queue holding pending tasks
        self.task_queue = Queue()
        self.populate_tasks()
        
        # Running tasks dictionary: {gpu_index: [Task, Task, ...]}
        self.running_tasks = {gpu: [] for gpu in gpu_indices}
        self.lock = threading.Lock()  # Protect shared data
        
        # Completed tasks
        self.completed_tasks = []
        
        # Add a flag to signal thread termination
        self.running = True
        
        # Start the logging thread
        self.log_thread = threading.Thread(target=self.update_main_log, daemon=True)
        self.log_thread.start()
        
        # Keep track of all container IDs for cleanup
        self.all_container_ids = []
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def populate_tasks(self):
        """Add all competition tasks to the queue without pre-assigning GPUs."""
        task_id = 0
        for competition in self.competitions:
            # Create a task without assigning a GPU index yet
            self.task_queue.put(Task(task_id=task_id, competition_name=competition))
            task_id += 1
        self.logger.info(f"Populated task queue with {task_id} tasks.")

    def launch_task(self, task):
        """Launch a task on its assigned GPU."""
        # Ensure task has a GPU assigned before launching (should be guaranteed by fill_tasks)
        if task.gpu_index is None:
             print(f"[{datetime.now()}] Error: Task {task.task_id} has no assigned GPU. Skipping launch.")
             return None

        # Fill in the docker run command with parameters. Use string representation of paths.
        cmd = DOCKER_RUN_TEMPLATE.format(
            gpu_index=task.gpu_index,
            competition_name=task.competition_name,
            docker_image=self.docker_image,
            main_py_path=str(self.main_py_path),
            config_yaml_path=str(self.config_yaml_path),
            agent_dir_path=str(self.agent_dir_path),
            kaggle_data=str(self.kaggle_data),
            output_dir=str(self.output_dir),
            env_path=str(self.env_path)
        )
        try:
            # Run the docker container in detached mode and capture its container id.
            container_id = subprocess.check_output(cmd, shell=True).decode().strip()
            task.container_id = container_id
            # Add container ID to our list for cleanup
            self.all_container_ids.append(container_id)
            task.start_time = datetime.now()
            
            # Set up container logging
            log_filename = f"{task.competition_name}_gpu{task.gpu_index}_{container_id[:12]}.log"
            task.log_file = self.container_logs_dir / log_filename
            
            # Start a thread to stream container logs
            task.log_thread = threading.Thread(
                target=self.stream_container_logs,
                args=(task.container_id, task.log_file),
                daemon=True
            )
            task.log_thread.start()
            
            self.logger.info(f"Launched task {task.task_id} (Comp: {task.competition_name}, GPU: {task.gpu_index}, Cont: {container_id[:12]})")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to launch task {task.task_id} (Comp: {task.competition_name}, GPU: {task.gpu_index}): {e}")
            return None
        return task
    
    def stream_container_logs(self, container_id, log_file):
        """Stream container logs to a file"""
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                log_process = subprocess.Popen(
                    ['docker', 'logs', '-f', container_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                for line in log_process.stdout:
                    # Decode the bytes to text with UTF-8 encoding
                    decoded_line = line.decode('utf-8', errors='replace')
                    f.write(decoded_line)
                    f.flush()
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nError capturing logs: {str(e)}\n")

    def update_main_log(self):
        """Update the main log file every 60 seconds"""
        while self.running:
            log_data = self.get_status_summary()
            log_path = self.log_dir / "all_log.json"
            
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error writing status JSON log: {e}")
                
            time.sleep(60)  # Update every 60 seconds

    def get_status_summary(self):
        """Get a summary of the current system status for logging"""
        with self.lock:
            running = []
            for gpu, tasks in self.running_tasks.items():
                for task in tasks:
                    running.append(task.to_dict())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "gpu_status": {
                    str(gpu): f"{len(tasks)}/{self.max_tasks_per_gpu} tasks running"
                    for gpu, tasks in self.running_tasks.items()
                },
                "running_tasks": running,
                "completed_tasks": [task.to_dict() for task in self.completed_tasks],
                "tasks_in_queue": self.task_queue.qsize()
            }

    def check_container_status(self, container_id):
        # Use docker inspect to get the container state. The format returns the status (running, exited, etc.)
        try:
            status = subprocess.check_output(
                f"docker inspect -f '{{{{.State.Status}}}}' {container_id}",
                shell=True
            ).decode().strip()
            return status
        except subprocess.CalledProcessError:
            # If inspect fails, assume the container no longer exists.
            return "not_found"

    def enforce_runtime(self, task):
        # Check if a task has exceeded its runtime, and kill if so.
        if task.start_time and datetime.now() - task.start_time > timedelta(seconds=self.task_runtime):
            try:
                subprocess.check_call(f"docker kill {task.container_id}", shell=True)
                self.logger.warning(f"Task {task.task_id} (Comp: {task.competition_name}, GPU: {task.gpu_index}, Cont: {task.container_id[:12]}) exceeded runtime ({self.task_runtime}s) and was killed.")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to kill timed-out task {task.task_id} (Comp: {task.competition_name}, Cont: {task.container_id[:12]}): {e}")

    def monitor_tasks(self):
        # Flag to ensure final log update happens only once
        final_log_written = False
        while True:
            all_finished = False # Reset check flag each iteration
            with self.lock:
                # For each GPU, check running tasks
                for gpu, tasks in self.running_tasks.items():
                    for task in tasks.copy(): 
                        status = self.check_container_status(task.container_id)
                        # Enforce runtime if still running
                        if status == "running":
                            self.enforce_runtime(task)
                        # If task is not running, assume it's completed or failed and remove it.
                        elif status != "running": # Use elif for clarity
                            self.logger.info(f"Task {task.task_id} (Comp: {task.competition_name}, GPU: {gpu}, Cont: {task.container_id[:12]}) finished with status: {status}")
                            self.completed_tasks.append(task)
                            tasks.remove(task) 

                # Check if all tasks are finished while still holding the lock
                all_finished = self.task_queue.empty() and all(len(ts) == 0 for ts in self.running_tasks.values())

            # Attempt to fill available GPU slots outside the main check lock
            # This allows filling to happen more promptly after a task finishes.
            if not all_finished:
                 self.fill_tasks()

            # Sleep for a short interval before next check
            time.sleep(5)

            # Exit loop if no tasks remain and no tasks are running
            if all_finished and not final_log_written:
                self.logger.info("All tasks completed.")
                # Final update to the main log before exiting - OUTSIDE the lock
                try:
                    log_data = self.get_status_summary()
                    log_path = self.log_dir / "all_log.json"
                    with open(log_path, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)
                    final_log_written = True 
                    self.logger.info(f"Final status summary written to {log_path}")
                except Exception as e:
                     self.logger.error(f"Error writing final status summary JSON: {e}")
                self.logger.info("Exiting monitor loop.")
                break
            elif all_finished and final_log_written:
                 # If already finished and logged, just break
                 break

    def fill_tasks(self):
        """Dynamically assign tasks from queue to available GPU slots."""
        # Fill all GPUs up to max_tasks_per_gpu if tasks remain in the queue.
        with self.lock: 
            for gpu in sorted(self.gpu_indices): 
                # While this GPU has capacity AND there are tasks waiting
                while len(self.running_tasks[gpu]) < self.max_tasks_per_gpu and not self.task_queue.empty():
                    # Get the next task from the queue
                    task = self.task_queue.get()

                    # Assign the current GPU to this task *dynamically*
                    task.gpu_index = gpu

                    # Attempt to launch the task
                    launched_task = self.launch_task(task) 

                    if launched_task:
                        # If launch succeeded, add to this GPU's running tasks
                        self.running_tasks[gpu].append(launched_task)
                    else:
                        # The launch_task method already logs the error
                        self.logger.warning(f"Launch failed for task {task.task_id} (Comp: {task.competition_name}) on GPU {gpu}. Task will not be retried automatically.")
                        # Consider adding failed task to a separate list or re-queuing if needed

    def handle_shutdown(self, signum, frame):
        """Signal handler for graceful shutdown"""
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up all resources before exiting"""
        self.logger.info("Starting cleanup process...")
        
        # Stop the main monitoring loop and logging thread
        self.running = False
        
        # Get all running container IDs launched by this program
        all_containers = set()
        with self.lock:
            # Add containers from running tasks
            for gpu, tasks in self.running_tasks.items():
                for task in tasks:
                    if task.container_id:
                        all_containers.add(task.container_id)
            
            # Add containers from completed tasks (in case any are still running)
            for task in self.completed_tasks:
                if task.container_id:
                    all_containers.add(task.container_id)
            
            # Add any containers we tracked separately
            all_containers.update(self.all_container_ids)
        
        # Kill all containers
        for container_id in all_containers:
            try:
                status = self.check_container_status(container_id)
                if status == "running":
                    self.logger.info(f"Killing container {container_id[:12]}...")
                    subprocess.run(["docker", "kill", container_id], 
                                  check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                self.logger.error(f"Error killing container {container_id[:12]}: {e}")
        
        # Wait for logging thread to finish
        if hasattr(self, 'log_thread') and self.log_thread.is_alive():
            self.logger.info("Waiting for status logging thread to finish...")
            self.log_thread.join(timeout=5) # Increased timeout slightly
        
        # Final log update
        try:
            log_data = self.get_status_summary()
            log_data["shutdown"] = "Program terminated via signal"
            log_path = self.log_dir / "all_log.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error writing final status summary JSON during cleanup: {e}")
        
        self.logger.info("Cleanup complete. Exiting.")
        logging.shutdown() # Explicitly shutdown logging

    def run(self):
        try:
            # Start the monitoring loop in the main thread.
            self.fill_tasks()
            self.monitor_tasks()
        finally:
            # Ensure cleanup happens even if an unhandled exception occurs
            self.cleanup()
        


if __name__ == "__main__":
    args = parse_args()
    
    # --- Logging Setup ---
    # Set up logging early, before creating the manager instance
    logger = setup_logging(args.log_dir)
    
    # Get competitions from file if specified, or use command line arguments
    competitions = []
    if args.competitions_file:
        # Resolve the competitions file path
        competitions_file_path = Path(args.competitions_file).resolve()
        if competitions_file_path.exists():
            try:
                with open(competitions_file_path, 'r') as f:
                    competitions = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(competitions)} competitions from {competitions_file_path}")
            except Exception as e:
                logger.error(f"Error reading competitions file {competitions_file_path}: {e}", exc_info=True) # Log traceback
        else:
            logger.warning(f"Competitions file not found at {competitions_file_path}. Using --competitions argument.")

    # If no competitions were found in the file or file wasn't specified, use the command line arguments
    if not competitions:
        competitions = args.competitions
        if not competitions:
             print(f"[{datetime.now()}] Error: No competitions specified either via --competitions or --competitions-file. Exiting.")
             sys.exit(1) # Exit if no competitions are provided

    # Pass resolved paths (or original strings to be resolved in __init__) and docker image to the manager
    manager = DockerTaskManager(
        gpu_indices=args.gpu_indices,
        max_tasks_per_gpu=args.max_tasks_per_gpu,
        task_runtime=args.task_runtime_seconds,
        competitions=competitions,
        log_dir=args.log_dir,
        kaggle_data=args.kaggle_data,
        output_dir=args.output_dir, 
        docker_image=args.docker_image 
    )
    manager.run()