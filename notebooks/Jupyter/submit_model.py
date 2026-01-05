import os
import subprocess

def run_command(command):
    """Runs a shell command and prints the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(e.stdout)
        print(e.stderr)

def main():
    """Main function to add, commit, and push changes."""
    print("Starting model submission...")

    # 1. Add all files
    print("\nStep 1: Adding all files...")
    run_command("git add .")

    # 2. Commit changes
    commit_message = "Submitting new model version"
    print(f"\nStep 2: Committing changes with message: '{commit_message}'...")
    run_command(f'git commit -m "{commit_message}"')

    # 3. Push changes
    print("\nStep 3: Pushing changes to the remote repository...")
    run_command("git push")

    print("\nModel submission complete!")

if __name__ == "__main__":
    main()

