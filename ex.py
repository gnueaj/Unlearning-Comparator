import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = 'b9e5'
JSON_FILE_PATH = f'/home/jaeung/mu-dashboard/backend/data/{FILE_NAME}.json'

def generate_confidence_heatmap(json_file_path):
    # JSON 파일 읽기
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # test confidence distribution 데이터 추출
    test_conf_dist = data['test_confidence_distribution']
    
    # 데이터를 numpy 배열로 변환
    conf_matrix = np.array([[test_conf_dist[f'gt_{i}'][f'pred_{j}'] for j in range(10)] for i in range(10)])
    
    # 히트맵 생성
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', fmt='.3f', 
                     xticklabels=range(10), yticklabels=range(10))
    
    plt.title('Test Confidence Distribution Heatmap')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    
    # 컬러바 레이블 추가
    cbar = ax.collections[0].colorbar
    cbar.set_label('Confidence')
    
    # 현재 디렉토리에 이미지 저장
    output_path = os.path.join(os.getcwd(), f'test_confidence_heatmap_{FILE_NAME}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved as {output_path}")

if __name__ == "__main__":
    generate_confidence_heatmap(JSON_FILE_PATH)