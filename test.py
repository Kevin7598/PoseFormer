import numpy as np
import sys

def view_npy(file_path, output_csv_path=None):
    """
    Loads a .npy file and prints its contents in a CSV-like format.
    
    Assumes the numpy array is 3D with shape (frames, keypoints, coordinates).
    """
    try:
        # Load the data from the .npy file
        data = np.load(file_path)
        print(f"Successfully loaded {file_path}", file=sys.stderr)
        print(f"Array Shape: {data.shape}", file=sys.stderr)
        print(f"Array Dtype: {data.dtype}", file=sys.stderr)
        print("-" * 30, file=sys.stderr)

        # Use a with statement to handle the file stream, which defaults to stdout
        output_stream = open(output_csv_path, 'w') if output_csv_path else sys.stdout
        with output_stream as f:
            if data.ndim != 3:
                print("Warning: Expected a 3D array (frames, keypoints, coords), but got a different shape.", file=sys.stderr)
                # Attempt to print it anyway
                for i, row in enumerate(data):
                    # Flatten each major element and print
                    print(f"{i},{','.join(map(str, np.ravel(row)))}", file=f)
                return

            num_frames, num_keypoints, num_coords = data.shape

            # Create and print a header
            header = ["frame_index"]
            for kp in range(num_keypoints):
                for coord_idx in range(num_coords):
                    header.append(f"kp{kp}_coord{coord_idx}")
            print(",".join(header), file=f)

            # Print each frame's data as a single CSV row
            for frame_idx, frame_data in enumerate(data):
                # Flatten the keypoints and coordinates for the current frame
                flat_data = frame_data.flatten()
                row_values = [str(frame_idx)] + [f"{val:.6f}" for val in flat_data]
                print(",".join(row_values), file=f)
        
        if output_csv_path:
            print(f"\nData successfully written to {output_csv_path}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # The path from your test.py file
    PATH = "/root/autodl-tmp/data/pose/train/B/train-00551.npy"
    OUTPUT_PATH = "/root/488/proj/posenet/output.csv"
    # To print to console, call: view_npy(PATH)
    # To save to a file, call:
    view_npy(PATH, OUTPUT_PATH)