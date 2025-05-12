from mledojo.competitions import get_prepare
from mledojo.competitions.utils import extract, download_competition_data
from pathlib import Path
import json
import os
import sys
import traceback
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare competition data')
    parser.add_argument('--competitions', nargs='+', help='List of competition names to prepare')
    parser.add_argument('--competitions-file', type=str, help='Path to a text file with competition names (one per line)')
    parser.add_argument('--data-dir', type=str, default='./data/prepared', help='Base directory for data')
    parser.add_argument('--logs-dir', type=str, default='./data/prepare_logs', help='Directory for results and logs')
    return parser.parse_args()

def get_competitions_from_file(file_path):
    """Read competition names from a text file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return []
    
    with open(file_path, 'r') as f:
        competitions = [line.strip() for line in f if line.strip()]
    
    return competitions

def prepare_competition(competition_name, data_dir, logs_dir):
    """Prepare data for a single competition."""
    # Convert paths to absolute paths
    data_dir = Path(data_dir).resolve()
    logs_dir = Path(logs_dir).resolve()
    
    # Get the project root directory (where mledojo package is located)
    project_root = Path(__file__).parent.parent.resolve()
    
    # Create log file for this competition
    log_file = logs_dir / f"{competition_name}.log"
    
    try:
        # Redirect stdout and stderr to the log file
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_f = open(log_file, "w")
        sys.stdout = log_f
        sys.stderr = log_f
        
        print(f"Processing competition: {competition_name}")
        
        data_dir_path = Path(data_dir)
        comp_dir = data_dir_path / competition_name

        # Create directory structure
        raw_data_dir = comp_dir / "raw"
        data_dir_comp = comp_dir / "data"
        public_data_dir = data_dir_comp / "public"
        private_data_dir = data_dir_comp / "private"

        for directory in [comp_dir, raw_data_dir, data_dir_comp, public_data_dir, private_data_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # Download and prepare data
        print("Downloading competition data...")
        download_competition_data(competition_name, raw_data_dir)
        print("Extracting data...")
        extract(raw_data_dir / f"{competition_name}.zip", raw_data_dir)
        print("Preparing data...")
        prepare = get_prepare(competition_name)
        prepare(raw_data_dir, public_data_dir, private_data_dir)

        # Use project root to find competition files
        mle_dojo_comp_dir = project_root / "mledojo" / "competitions" / competition_name
        
        # Copy description.txt to public_data_dir
        description_file = mle_dojo_comp_dir / "info" / "description.txt"
        if description_file.exists():
            shutil.copy2(description_file, public_data_dir / "description.txt")
            print(f"Copied description.txt to {public_data_dir}")
        else:
            print(f"Warning: description.txt not found in {mle_dojo_comp_dir}/info")
        
        # Copy leaderboard files to private_data_dir
        info_dir = mle_dojo_comp_dir / "info"
        if info_dir.exists():
            # Copy public_leaderboard.csv
            public_leaderboard = info_dir / "public_leaderboard.csv"
            if public_leaderboard.exists():
                shutil.copy2(public_leaderboard, private_data_dir / "public_leaderboard.csv")
                print(f"Copied public_leaderboard.csv to {private_data_dir}")
            else:
                print(f"Warning: public_leaderboard.csv not found in {info_dir}")
            
            # Copy private_leaderboard.csv
            private_leaderboard = info_dir / "private_leaderboard.csv"
            if private_leaderboard.exists():
                shutil.copy2(private_leaderboard, private_data_dir / "private_leaderboard.csv")
                print(f"Copied private_leaderboard.csv to {private_data_dir}")
            else:
                print(f"Warning: private_leaderboard.csv not found in {info_dir}")
        else:
            print(f"Warning: Info directory not found for {competition_name}")
        
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_f.close()
        
        return True, None
            
    except Exception as e:
        # Restore stdout and stderr
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_f.close()
        
        # Log the error
        with open(log_file, "a") as f:
            f.write(f"Error processing {competition_name}: {e}\n")
            f.write(traceback.format_exc())
        
        return False, str(e)

def prepare_competitions(competitions, data_dir, results_dir):
    """Prepare data for multiple competitions."""
    # Convert paths to absolute paths
    results_path = Path(results_dir).resolve()
    data_path = Path(data_dir).resolve()

    # Create results directory if it doesn't exist
    results_path.mkdir(exist_ok=True, parents=True)

    # Create data directory if it doesn't exist
    data_path.mkdir(exist_ok=True, parents=True)

    # Create log directories
    success_dir = results_path / "success"
    failed_dir = results_path / "failed"

    # Create .txt files for success and failed competitions
    success_file = results_path / "success.txt"
    failed_file = results_path / "failed.txt"
    success_file.touch(exist_ok=True)
    failed_file.touch(exist_ok=True)

    for directory in [results_path, success_dir, failed_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    success = []
    failed = []
    
    for competition_name in competitions:
        print(f"Preparing competition: {competition_name}")
        success_flag, error_msg = prepare_competition(competition_name, data_path, results_path)
        
        log_file = results_path / f"{competition_name}.log"
        
        if success_flag:
            success.append(competition_name)
            # Copy log to success directory for easy access
            shutil.copy2(log_file, success_dir / f"{competition_name}.log")
            with open(success_file, "a") as f:
                f.write(f"{competition_name}\n")
            print(f"Successfully prepared {competition_name}")
        else:
            failed.append(competition_name)
            # Copy log to failed directory for easy access
            shutil.copy2(log_file, failed_dir / f"{competition_name}.log")
            with open(failed_file, "a") as f:
                f.write(f"{competition_name}\n")
            print(f"Failed to prepare {competition_name}: {error_msg}")
            
            # Delete the competition directory if preparation failed
            comp_dir = data_path / competition_name
            if comp_dir.exists():
                print(f"Removing failed competition directory: {comp_dir}")
                shutil.rmtree(comp_dir, ignore_errors=True)
        
        # Remove the original log file since we've copied it to success or failed directory
        os.remove(log_file)
    
    print(f"Successfully processed competitions: {success}")
    print(f"Failed competitions: {failed}")
    
    return success, failed

def main():
    args = parse_args()
    
    # Convert data_dir and logs_dir to absolute paths if they're relative
    data_dir = Path(args.data_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.resolve()
    
    # Determine which competitions to prepare
    competitions = []
    
    if args.competitions_file:
        competitions = get_competitions_from_file(args.competitions_file)
        print(f"Loaded {len(competitions)} competitions from file {args.competitions_file}")
    elif args.competitions:
        competitions = args.competitions
        print(f"Preparing {len(competitions)} competitions from command line arguments")
    else:
        # Use project root to find competition.json
        competition_json_path = project_root / "mledojo" / "competitions" / "competition.json"
        
        if not competition_json_path.exists():
            print(f"Error: competition.json not found at {competition_json_path}")
            return
            
        with open(competition_json_path, "r") as f:
            data = json.load(f)
        
        MLE_LITE = data["Evaluation"]["MLE-Lite"]
        NLP = data["Evaluation"]["NLP"]
        CV = data["Evaluation"]["CV"]
        Tabular = data["Evaluation"]["Tabular"]
        Training = data["Training"]
        
        competitions = MLE_LITE + NLP + CV + Tabular + Training
        print(f"No competitions specified, preparing all {len(competitions)} competitions")
    
    # Ensure data and logs directories exist
    data_dir.mkdir(exist_ok=True, parents=True)
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare the competitions
    prepare_competitions(competitions, data_dir, logs_dir)

if __name__ == "__main__":
    main()