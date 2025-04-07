import pandas as pd
import numpy as np
import os

def get_distribution(file_path):
    """CSV 파일의 감정 분포를 반환"""
    df = pd.read_csv(file_path)
    counts = df['emotion'].value_counts().sort_index()
    total = len(df)
    percentages = (counts / total * 100).round(2)
    return counts, percentages

def compare_fold(fold_num, gender):
    """특정 fold와 성별에 대한 분포 비교"""
    print(f"\n=== Comparing Fold {fold_num} {gender} ===")
    
    # IEMOCAP 분포
    iemocap_test = f"iemocap/iemocap_{fold_num:02d}{gender}.test.csv"
    iemocap_train = f"iemocap/iemocap_{fold_num:02d}{gender}.train.csv"
    
    iemocap_test_counts, iemocap_test_pcts = get_distribution(iemocap_test)
    iemocap_train_counts, iemocap_train_pcts = get_distribution(iemocap_train)
    
    print("\nIEMOCAP Test Distribution:")
    for emotion in iemocap_test_counts.index:
        print(f"{emotion}: {iemocap_test_counts[emotion]} samples ({iemocap_test_pcts[emotion]}%)")
    
    print("\nIEMOCAP Train Distribution:")
    for emotion in iemocap_train_counts.index:
        print(f"{emotion}: {iemocap_train_counts[emotion]} samples ({iemocap_train_pcts[emotion]}%)")
    
    # KEMDy19 분포
    kemdy_test = f"kemdy19/kemdy19_fold{fold_num:02d}{gender}.test.csv"
    kemdy_train = f"kemdy19/kemdy19_fold{fold_num:02d}{gender}.train.csv"
    
    kemdy_test_counts, kemdy_test_pcts = get_distribution(kemdy_test)
    kemdy_train_counts, kemdy_train_pcts = get_distribution(kemdy_train)
    
    print("\nKEMDy19 Test Distribution:")
    for emotion in kemdy_test_counts.index:
        print(f"{emotion}: {kemdy_test_counts[emotion]} samples ({kemdy_test_pcts[emotion]}%)")
    
    print("\nKEMDy19 Train Distribution:")
    for emotion in kemdy_train_counts.index:
        print(f"{emotion}: {kemdy_train_counts[emotion]} samples ({kemdy_train_pcts[emotion]}%)")
    
    # 분포가 정확히 일치하는지 확인
    test_match = all(iemocap_test_counts == kemdy_test_counts)
    train_match = all(iemocap_train_counts == kemdy_train_counts)
    
    print(f"\nDistribution Match:")
    print(f"Test: {'✓' if test_match else '✗'}")
    print(f"Train: {'✓' if train_match else '✗'}")

def main():
    num_folds = 5
    genders = ['F', 'M']
    
    for fold in range(1, num_folds + 1):
        for gender in genders:
            compare_fold(fold, gender)

if __name__ == "__main__":
    main() 