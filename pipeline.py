import subprocess
import sys
from datetime import datetime
from pathlib import Path

def log_message(message, log_file):
    """Write a message to the log file with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def run_script(script_name, log_file):
    """Run a Python script and log its output."""
    log_message(f"Starting {script_name}...", log_file)
    try:
        # Run script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        # Log stdout
        if result.stdout:
            log_message(result.stdout, log_file)
        log_message(f"{script_name} completed successfully.", log_file)
        return True
    except subprocess.CalledProcessError as e:
        # Log error and stderr
        log_message(f"Error in {script_name}: {e}", log_file)
        if e.stderr:
            log_message(e.stderr, log_file)
        return False
    except Exception as e:
        # Log unexpected errors
        log_message(f"Unexpected error in {script_name}: {e}", log_file)
        return False

def main():
    log_file = Path("pipeline_log.txt")
    scripts = ["get_quotations.py", "process_data.py", "get_prediction.py", "x_post.py"]

    # Ensure log file exists
    log_file.touch(exist_ok=True)

    log_message("=== Starting Pipeline ===", log_file)

    for script in scripts:
        if not Path(script).exists():
            log_message(f"Error: {script} not found!", log_file)
            sys.exit(1)
        
        success = run_script(script, log_file)
        if not success:
            log_message(f"Pipeline stopped due to error in {script}.", log_file)
            sys.exit(1)
        
        log_message(f"--- Finished {script} ---", log_file)

    log_message("=== Pipeline Completed Successfully ===", log_file)

if __name__ == "__main__":
    main()