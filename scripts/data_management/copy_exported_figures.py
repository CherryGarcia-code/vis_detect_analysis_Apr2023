import pandas as pd
import shutil
import os
from pathlib import Path
import logging

def main():
    csv_path = 'photometry_export_matlab.csv'
    out_dir = Path('exported_figures')
    out_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if not os.path.exists(csv_path):
        logging.error(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Get unique sessions
    pairs = df[['subject_id', 'session_id']].drop_duplicates()
    
    logging.info(f"Found {len(pairs)} unique sessions to process.")
    
    copied_count = 0
    missing_count = 0
    
    for _, row in pairs.iterrows():
        subj = str(row['subject_id'])
        sess_id = str(row['session_id'])
        
        # Parse components
        try:
            # handle BG_013 -> 013
            parts = subj.split('_')
            if len(parts) > 1:
                sub_num = parts[-1]
            else:
                sub_num = subj # Fallback
                
            # handle 20240118_121004 -> 20240118
            date_str = sess_id.split('_')[0]
            
            # Construct File Path
            # Format: FIGURES/{Subject}/{SubjectNum}_{Date}/Outcome_Comparison.png
            folder_name = f"{sub_num}_{date_str}"
            
            # Try Standard Path
            src_path_1 = Path("FIGURES") / subj / folder_name / "Outcome_Comparison.png"
            
            # Identify source
            final_src = None
            if src_path_1.exists():
                final_src = src_path_1
            else:
                # Try finding folder regardless of prefix?
                # Sometimes folder might be just Date? Or BG included?
                # Let's check parent dir
                parent_dir = Path("FIGURES") / subj
                if parent_dir.exists():
                    # Look for folder containing the date
                    candidates = [d for d in parent_dir.iterdir() if d.is_dir() and date_str in d.name]
                    if candidates:
                        # Pick first
                        pot_path = candidates[0] / "Outcome_Comparison.png"
                        if pot_path.exists():
                            final_src = pot_path

            if final_src:
                dst_name = f"{subj}_{date_str}_Outcome_Comparison.png"
                dst_path = out_dir / dst_name
                shutil.copy(final_src, dst_path)
                copied_count += 1
            else:
                logging.warning(f"Missing Figure: {subj} {date_str} (Checked {src_path_1})")
                missing_count += 1
                
        except Exception as e:
            logging.error(f"Error processing {subj} {sess_id}: {e}")
            missing_count += 1

    logging.info(f"Finished. Copied: {copied_count}, Missing: {missing_count}")
    logging.info(f"Files saved to: {out_dir.absolute()}")

if __name__ == "__main__":
    main()
