#!/usr/bin/env python3
"""
주어진 디렉토리(/home/jovyan/YJ_DATA/interspeech21_emotion/iemocap) 내의 모든 CSV 파일을 찾아,
각 파일의 첫 줄(헤더)을 제외한 row의 총합을 계산하는 스크립트입니다.
"""

import os
import glob

def count_csv_rows(directory):
    total_rows = 0
    # 디렉토리 내 모든 서브디렉토리를 포함해 csv 파일들을 찾습니다.
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    # 첫번째 줄은 헤더이므로 제외
                    num_rows = len(lines) - 1
                    total_rows += num_rows
        except Exception as e:
            print(f"{csv_file} 파일 처리 중 예외 발생: {e}")
    
    return total_rows

if __name__ == '__main__':
    directory = '/home/jovyan/YJ_DATA/interspeech21_emotion/iemocap'
    total_csv_rows = count_csv_rows(directory)
    print("헤더 제외 총 CSV row 수:", total_csv_rows)
