from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 모델 confusion matrix 평가 함수
def evaluate_model(y_test, y_pred, model_name):
    """
    주어진 모델의 성능을 평가하고 혼동 행렬과 분류 보고서를 출력합니다.

    Parameters:
    y_test (array-like): 테스트 데이터의 실제 레이블
    y_pred (array-like): 모델이 예측한 레이블
    model_name (str): 모델의 이름 또는 설명

    Returns:
    None
    """
    class_names = [f'Class {i}' for i in range(4)]

    print(f"\n{model_name} 모델 평가")
    # 분류 보고서 출력
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 혼동 행렬 출력
    print("혼동 행렬:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
