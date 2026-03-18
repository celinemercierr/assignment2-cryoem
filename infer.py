#!/usr/bin/env python3
import os
import pandas as pd
import torch
from model import MicrographCleaner
from dataset import InferenceMicrographDataset, decode_array, encode_array
from inference_utils import sliding_window_inference
import matplotlib.pyplot as plt
import tqdm
 
 
def main():
    # Create predictions directory
    os.makedirs('predictions', exist_ok=True)
 
    # Parameters
    WINDOW_SIZE = 512
    THRESHOLD = 0.5
    OVERLAP = 0.5
 
    # ------------------------------------------------------------------ #
    # Load model
    # NOTE: train.py saves "final_checkpoint.pt" — must match here.
    # ------------------------------------------------------------------ #
    model = MicrographCleaner.load_from_checkpoint(
        'final_checkpoint.pt', map_location='cpu'
    )
    model.eval()
 
    # Load test data
    test_df = pd.read_csv('test.csv')
    test_dataset = InferenceMicrographDataset(test_df, window_size=WINDOW_SIZE)
 
    # We'll collect predictions to build the Kaggle submission CSV too.
    submission_rows = []
 
    with torch.inference_mode():
        # -------------------------------------------------------------- #
        # IMPORTANT: iterate over every row — no deduplication.
        # evaluate.sh requires exactly one PNG per row in test.csv.
        # -------------------------------------------------------------- #
        for idx in tqdm.tqdm(range(len(test_dataset))):
            image, image_id, (pad_h, pad_w) = test_dataset[idx]
 
            # Sliding-window inference on the full image
            pred = sliding_window_inference(
                model,
                image,
                window_size=WINDOW_SIZE,
                overlap=OVERLAP,
            )
 
            # Remove padding added during preprocessing
            if pad_h > 0:
                pred = pred[..., :-pad_h, :]
            if pad_w > 0:
                pred = pred[..., :-pad_w]
 
            # Binary mask
            pred_mask = (pred > THRESHOLD).float().cpu().numpy()[0]
 
            # ---------------------------------------------------------- #
            # Visualisation — side-by-side original + predicted mask
            # Use idx to fetch the correct row from test_df.
            # ---------------------------------------------------------- #
            orig_image = decode_array(test_df.iloc[idx]['image'])
 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(orig_image, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
 
            ax2.imshow(pred_mask, cmap='gray')
            ax2.set_title('Predicted Mask')
            ax2.axis('off')
 
            plt.tight_layout()
 
            # Use row-index prefix so filenames are always unique,
            # even if the same image_id appears in multiple rows.
            png_path = f'predictions/{idx:05d}_{image_id}.png'
            plt.savefig(png_path, dpi=100)
            plt.close()
 
            # Encode mask for Kaggle submission CSV
            submission_rows.append({
                'Id': image_id,
                'mask': encode_array(pred_mask.astype('float32')),
            })
 
    # Save Kaggle submission CSV
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nDone. {len(submission_rows)} predictions saved.")
    print(f"  PNGs  → predictions/")
    print(f"  Kaggle → submission.csv")
 
 
if __name__ == "__main__":
    main()