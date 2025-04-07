import pandas as pd
import numpy as np
import argparse
import os
import glob

# 재현성을 위한 random seed 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def get_iemocap_sizes(fold_num, gender):
    """IEMOCAP의 특정 fold와 성별에 대한 train/test 크기를 반환"""
    test_file = f"iemocap/iemocap_{fold_num:02d}{gender}.test.csv"
    train_file = f"iemocap/iemocap_{fold_num:02d}{gender}.train.csv"
    
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    
    test_sizes = test_df['emotion'].value_counts().sort_index().to_dict()
    train_sizes = train_df['emotion'].value_counts().sort_index().to_dict()
    
    return test_sizes, train_sizes

def balance_dataset(input_file, output_file, target_sizes):
    """데이터셋을 target_sizes에 맞게 조정"""
    df = pd.read_csv(input_file)
    
    balanced_dfs = []
    sampling_info = {}  # 샘플링 정보 저장
    
    for emotion, target_size in target_sizes.items():
        emotion_df = df[df['emotion'] == emotion]
        
        if len(emotion_df) > target_size:
            # 무작위로 target_size만큼 샘플링
            sampled_df = emotion_df.sample(n=target_size, random_state=RANDOM_SEED)
        else:
            # 데이터가 부족한 경우 중복 샘플링
            sampled_df = emotion_df.sample(n=target_size, replace=True, random_state=RANDOM_SEED)
            
        balanced_dfs.append(sampled_df)
        
        # 샘플링 정보 저장
        sampling_info[emotion] = {
            'original_size': len(emotion_df),
            'target_size': target_size,
            'sampled_files': sampled_df['file'].tolist()
        }
    
    # 모든 감정의 데이터를 합치기
    balanced_df = pd.concat(balanced_dfs)
    
    # 결과를 섞기
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    balanced_df.to_csv(output_file, index=False)
    
    # 샘플링 정보 저장
    info_file = output_file.replace('.csv', '.info.json')
    pd.DataFrame(sampling_info).to_json(info_file)
    
    # 결과 출력
    print(f"\n=== Balanced Dataset ({os.path.basename(output_file)}) ===")
    value_counts = balanced_df['emotion'].value_counts().sort_index()
    for emotion, count in value_counts.items():
        print(f"{emotion}: {count} samples")
        print(f"Original size: {sampling_info[emotion]['original_size']}")

def main():
    num_folds = 5
    genders = ['F', 'M']
    
    for fold in range(1, num_folds + 1):
        for gender in genders:
            print(f"\nProcessing fold {fold} {gender}...")
            
            # IEMOCAP 크기 가져오기
            test_sizes, train_sizes = get_iemocap_sizes(fold, gender)
            
            # KEMDy19 데이터셋 조정
            input_test = f"kemdy19/kemdy19_fold{fold:02d}.test.csv"
            input_train = f"kemdy19/kemdy19_fold{fold:02d}.train.csv"
            output_test = f"kemdy19_balanced/kemdy19_fold{fold:02d}{gender}.test.csv"
            output_train = f"kemdy19_balanced/kemdy19_fold{fold:02d}{gender}.train.csv"
            
            balance_dataset(input_test, output_test, test_sizes)
            balance_dataset(input_train, output_train, train_sizes)

if __name__ == "__main__":
    main() 