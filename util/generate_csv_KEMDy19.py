import os
import csv
import glob

KEMDY19_ROOT = "/home/YJ_DATA/dataset/KEMDy19_v1_3"
WAV_ROOT = os.path.join(KEMDY19_ROOT, "wav")
ANNOTATION_ROOT = os.path.join(KEMDY19_ROOT, "annotation")
OUTPUT_DIR = "/home/YJ_DATA/interspeech21_emotion/kemdy19"

emotion_map = {
    "neutral": "e0",
    "happy": "e1",
    "angry": "e2",
    "sad": "e3",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_annotation_csv(csv_path):
    segments = []
    
    import pandas as pd

    df = pd.read_csv(csv_path, header=[0, 1], encoding='utf-8')
    
    def format_column(col):
        if 'Unnamed' in col[0] and 'Unnamed' in col[1]:
            return ""
        elif 'Unnamed' in col[0]:
            return col[1]
        elif 'Unnamed' in col[1]:
            return col[0]
        else:
            return f"{col[0]}_{col[1]}"

    df.columns = [format_column(col) for col in df.columns]
    
    basename = os.path.basename(csv_path)
    session_str = basename.split('_')[0]
    session_num = int(session_str.replace("Session", ""))
    
    for i, row in df.iterrows():
        seg_id = row['Segment ID']
        emo = row['Total Evaluation_Emotion']
        
        if isinstance(emo, str) and ';' in emo:
            emo = emo.split(';')[0].strip()
        
        if emo not in emotion_map:
            continue
        
        emo = emotion_map[emo]
            
        parts = seg_id.split('_')
        dir_name = '_'.join(parts[:2])
        txt_name = '_'.join(parts[:3]) + ".txt"
        txt_path = os.path.join(
            WAV_ROOT,
            f"Session{session_num:02d}",
            dir_name,
            txt_name
        )
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""
            
        segments.append((seg_id, emo, text))
        
    return segments

def make_csv_lines_for_session(session_num, gender):
    all_lines = []
    csv_pattern = os.path.join(ANNOTATION_ROOT, f"Session{session_num:02d}_{gender}_res.csv")
    ann_files = glob.glob(csv_pattern)
    
    for ann_csv in ann_files:
        segments = parse_annotation_csv(ann_csv)
        
        for seg_id, emo, txt in segments:
            parts = seg_id.split('_')
            dir_name = '_'.join(parts[:2])
            wave_name = '_'.join(parts[:3]) + ".wav"
            wave_path = os.path.join(
                WAV_ROOT,
                f"Session{session_num:02d}",
                dir_name,
                wave_name
            )
            all_lines.append((wave_path, emo, txt))
            
    return all_lines

def main():
    num_sessions = 20
    num_folds = 10
    session_gender_data = {}
    
    for s in range(1, num_sessions+1):
        for gender in ['F', 'M']:
            data_lines = make_csv_lines_for_session(s, gender)
            session_gender_data[(s, gender)] = data_lines
            
    for fold in range(num_folds):
        test_sessions = [fold*2 + 1, fold*2 + 2]
        test_data = []
        train_data = []
        
        for test_sess in test_sessions:
            for test_gender in ['F', 'M']:
                test_data.extend(session_gender_data[(test_sess, test_gender)])
        
        for s in range(1, num_sessions+1):
            for g in ['F', 'M']:
                if s not in test_sessions:
                    train_data.extend(session_gender_data[(s, g)])
        
        out_test_csv = os.path.join(OUTPUT_DIR, f"kemdy19_fold{fold+1:02d}.test.csv")
        out_train_csv = os.path.join(OUTPUT_DIR, f"kemdy19_fold{fold+1:02d}.train.csv")
        
        with open(out_test_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "emotion", "text"])
            for (wav_path, emo, txt) in test_data:
                writer.writerow([wav_path, emo, txt])
            
        with open(out_train_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "emotion", "text"])
            for (wav_path, emo, txt) in train_data:
                writer.writerow([wav_path, emo, txt])
            
        print(f"[Fold {fold+1:02d}] train: {len(train_data)} / test: {len(test_data)}")

if __name__ == "__main__":
    main()
