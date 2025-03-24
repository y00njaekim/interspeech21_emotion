import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_emotion_distribution(test_file, train_file):
    # 데이터 로드
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    
    # 각 데이터셋의 감정 레이블 카운트
    test_counts = test_df['emotion'].value_counts().sort_index()
    train_counts = train_df['emotion'].value_counts().sort_index()
    
    # 전체 데이터셋 감정 레이블 카운트
    total_counts = pd.concat([test_df, train_df])['emotion'].value_counts().sort_index()
    
    # 퍼센티지 계산
    test_percent = (test_counts / len(test_df) * 100).round(2)
    train_percent = (train_counts / len(train_df) * 100).round(2)
    total_percent = (total_counts / len(total_counts) * 100).round(2)
    
    # 결과 출력
    print("\n=== Test Dataset ===")
    for emotion, count in test_counts.items():
        print(f"{emotion}: {count} samples ({test_percent[emotion]}%)")
        
    print("\n=== Train Dataset ===")
    for emotion, count in train_counts.items():
        print(f"{emotion}: {count} samples ({train_percent[emotion]}%)")
        
    print("\n=== Total Dataset ===")
    for emotion, count in total_counts.items():
        print(f"{emotion}: {count} samples ({total_percent[emotion]}%)")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # Test Dataset
    plt.subplot(131)
    sns.barplot(x=test_counts.index, y=test_counts.values)
    plt.title('Test Dataset')
    plt.xlabel('Emotion Label')
    plt.ylabel('Count')
    
    # Train Dataset
    plt.subplot(132)
    sns.barplot(x=train_counts.index, y=train_counts.values)
    plt.title('Train Dataset')
    plt.xlabel('Emotion Label')
    plt.ylabel('Count')
    
    # Total Dataset
    plt.subplot(133)
    sns.barplot(x=total_counts.index, y=total_counts.values)
    plt.title('Total Dataset')
    plt.xlabel('Emotion Label')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('emotion_distribution.png')
    plt.close()

if __name__ == "__main__":
    test_file = "../kemdy19/kemdy19_fold00.test.csv"
    train_file = "../kemdy19/kemdy19_fold00.train.csv"
    analyze_emotion_distribution(test_file, train_file) 