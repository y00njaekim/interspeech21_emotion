import pandas as pd
import numpy as np

def sample_kemdy19_data(input_test_file, input_train_file, output_test_file, output_train_file):
    # IEMOCAP 타겟 샘플 수
    test_targets = {
        'e0': 226,
        'e1': 117,
        'e2': 97,
        'e3': 67
    }
    
    train_targets = {
        'e0': 1482,
        'e1': 1519,
        'e2': 1006,
        'e3': 1017
    }
    
    # 데이터 로드
    test_df = pd.read_csv(input_test_file)
    train_df = pd.read_csv(input_train_file)
    
    # 테스트 데이터 샘플링
    sampled_test_dfs = []
    for emotion, target_count in test_targets.items():
        emotion_df = test_df[test_df['emotion'] == emotion]
        if len(emotion_df) >= target_count:
            sampled = emotion_df.sample(n=target_count, random_state=42)
        else:
            # 부족한 경우 중복 허용 샘플링
            sampled = emotion_df.sample(n=target_count, replace=True, random_state=42)
        sampled_test_dfs.append(sampled)
    
    # 학습 데이터 샘플링
    sampled_train_dfs = []
    for emotion, target_count in train_targets.items():
        emotion_df = train_df[train_df['emotion'] == emotion]
        if len(emotion_df) >= target_count:
            sampled = emotion_df.sample(n=target_count, random_state=42)
        else:
            # 부족한 경우 중복 허용 샘플링
            sampled = emotion_df.sample(n=target_count, replace=True, random_state=42)
        sampled_train_dfs.append(sampled)
    
    # 샘플링된 데이터 병합
    final_test_df = pd.concat(sampled_test_dfs, ignore_index=True)
    final_train_df = pd.concat(sampled_train_dfs, ignore_index=True)
    
    # 결과 저장
    final_test_df.to_csv(output_test_file, index=False)
    final_train_df.to_csv(output_train_file, index=False)
    
    # 결과 출력
    print(f"\n=== Sampled Test Dataset ({len(final_test_df)} samples) ===")
    test_counts = final_test_df['emotion'].value_counts().sort_index()
    for emotion, count in test_counts.items():
        print(f"{emotion}: {count} samples")
    
    print(f"\n=== Sampled Train Dataset ({len(final_train_df)} samples) ===")
    train_counts = final_train_df['emotion'].value_counts().sort_index()
    for emotion, count in train_counts.items():
        print(f"{emotion}: {count} samples")

if __name__ == "__main__":
    input_test_file = "../kemdy19/kemdy19_fold00.test.csv"
    input_train_file = "../kemdy19/kemdy19_fold00.train.csv"
    output_test_file = "../kemdy19/kemdy19_fold00.sampled.test.csv"
    output_train_file = "../kemdy19/kemdy19_fold00.sampled.train.csv"
    
    sample_kemdy19_data(input_test_file, input_train_file, output_test_file, output_train_file) 