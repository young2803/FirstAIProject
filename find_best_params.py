import json
from collections import Counter

def find_generalized_hyperparams(file_path, threshold=0.1):
    best_mse_list = []
    params_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            best_mse_list.append(data['best_mse'])
            params_list.append(tuple(data['best_params'].items()))  # 튜플로 변환해 비교 가능하게 저장
    
    # 최적 MSE 기준을 설정 (상위 threshold 비율 이내의 값들만 고려)
    best_mse_min = min(best_mse_list)
    mse_threshold = best_mse_min + (max(best_mse_list) - best_mse_min) * threshold
    
    filtered_params = [params_list[i] for i in range(len(best_mse_list)) if best_mse_list[i] <= mse_threshold]
    
    # 가장 많이 등장한 하이퍼파라미터 조합 찾기
    most_common_params = Counter(filtered_params).most_common(1)[0][0]
    best_params = dict(most_common_params)  # 다시 딕셔너리로 변환
    
    return best_params

# 사용 예시
file_path = "model_results.json"  # 파일 경로 설정
best_params = find_generalized_hyperparams(file_path)
print(f"Generalized Best Params: {best_params}")