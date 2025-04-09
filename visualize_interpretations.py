import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from pathlib import Path
from collections import defaultdict

def visualize_interpretations(input_dir, output_dir, num_samples=None, vis_type='both'):
    """
    Loads interpretation results (.npy files) and visualizes them as plots.

    Args:
        input_dir (str): Directory containing the .npy files (_attention_weights.npy, _ig_feature_extractor.npy).
        output_dir (str): Directory to save the visualization plots.
        num_samples (int, optional): Number of samples to visualize. If None, visualize all. Defaults to None.
        vis_type (str, optional): Type of visualization ('attention', 'ig', 'both'). Defaults to 'both'.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization type: {vis_type}")

    # Find all .npy files and group them by base filename
    npy_files = glob.glob(os.path.join(input_dir, '*.npy'))
    grouped_files = defaultdict(dict)
    for f in npy_files:
        path = Path(f)
        if path.stem.endswith('_attention_weights'):
            base_name = path.stem.replace('_attention_weights', '')
            grouped_files[base_name]['attention'] = str(path)
        elif path.stem.endswith('_ig_feature_extractor'):
            base_name = path.stem.replace('_ig_feature_extractor', '')
            grouped_files[base_name]['ig'] = str(path)

    print(f"Found {len(grouped_files)} samples.")

    # Select samples to visualize
    sample_keys = list(grouped_files.keys())
    if num_samples is not None and num_samples < len(sample_keys):
        # Optionally, sort keys or select randomly if needed
        sample_keys = sample_keys[:num_samples]
        print(f"Visualizing first {num_samples} samples.")
    else:
        print(f"Visualizing all {len(sample_keys)} samples.")


    for base_name in sample_keys:
        files = grouped_files[base_name]
        
        att_file = files.get('attention')
        ig_file = files.get('ig')

        # Determine number of subplots needed
        num_plots = 0
        if vis_type in ['attention', 'both'] and att_file:
            num_plots += 1
        if vis_type in ['ig', 'both'] and ig_file:
            num_plots += 1
            
        if num_plots == 0:
            print(f"Skipping {base_name}: No required .npy files found for vis_type '{vis_type}'.")
            continue

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), squeeze=False)
        plot_idx = 0
        title = f"Interpretation: {base_name}"

        # Plot Attention Weights
        if vis_type in ['attention', 'both'] and att_file:
            try:
                attention_weights = np.load(att_file)
                ax = axes[plot_idx, 0]
                ax.plot(attention_weights)
                ax.set_title('Attention Weights')
                ax.set_xlabel('Encoder Frame Index (Hidden State)')
                ax.set_ylabel('Weight')
                ax.grid(True)
                plot_idx += 1
            except Exception as e:
                print(f"Error loading or plotting {att_file}: {e}")
                title += f" (Error plotting Attention)"


        # Plot Integrated Gradients (Feature Extractor Output)
        if vis_type in ['ig', 'both'] and ig_file:
            try:
                ig_map = np.load(ig_file)
                ax = axes[plot_idx, 0]
                ax.plot(ig_map)
                ax.set_title('Integrated Gradients (Feature Extractor Output)')
                ax.set_xlabel('Feature Extractor Frame Index')
                ax.set_ylabel('Attribution (Summed Absolute)')
                ax.grid(True)
                plot_idx += 1
            except Exception as e:
                print(f"Error loading or plotting {ig_file}: {e}")
                title += f" (Error plotting IG)"
        
        fig.suptitle(title, y=1.02) # Adjust title position slightly
        fig.tight_layout() # Adjust layout to prevent overlap

        output_filename = os.path.join(output_dir, f"{base_name}_interpretation.png")
        try:
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"Saved plot: {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize interpretation results (.npy files).")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the .npy interpretation files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the visualization plots.")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to visualize (default: all).")
    parser.add_argument("--vis_type", type=str, default='both', choices=['attention', 'ig', 'both'],
                        help="Type of visualization ('attention', 'ig', 'both').")
    
    args = parser.parse_args()
    
    visualize_interpretations(args.input_dir, args.output_dir, args.num_samples, args.vis_type) 