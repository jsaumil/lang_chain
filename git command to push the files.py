import subprocess
import time
import os

# Define the commit message
commit_message = "Updated files"

# Define size limit (1GB = 1 * 1024 * 1024 * 1024 bytes)
SIZE_LIMIT = 90 * 1024 * 1024

def get_file_sizes():
    """Get a list of files and their sizes in the repository."""
    files = subprocess.run(["git", "ls-files", "-o", "-m","--exclude-standard"], capture_output=True, text=True).stdout.splitlines()
    file_sizes = [(f, os.path.getsize(f)) for f in files if os.path.exists(f)]
    return file_sizes

def stage_limited_files():
    """Stage only up to 1GB of files."""
    file_sizes = get_file_sizes()
    total_size = 0
    staged_files = []
    
    for file, size in file_sizes:
        if total_size + size > SIZE_LIMIT:
            break
        subprocess.run(["git", "add", file], check=True)
        staged_files.append(file)
        total_size += size
    
    return staged_files, total_size

def pull_changes():
    try:
        print("Pulling the latest changes from the repository...")
        subprocess.run(["git", "pull"], check=True)
        print("Successfully pulled the latest changes.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pulling: {e}")

def push_changes():
    try:
        pull_changes()
        
        staged_files, total_size = stage_limited_files()
        if not staged_files:
            print("No files to commit within the 1GB limit.")
            return
        
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print(f"Successfully pushed {len(staged_files)} files ({total_size / (1024 * 1024)} MB).")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

try:
    print("Auto-push started. Press Ctrl+C to stop.")
    while True:
        push_changes()
        time.sleep(30)  # Wait for 30 seconds before the next push
except KeyboardInterrupt:
    print("Auto-push stopped.")
