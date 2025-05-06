from AT_AFO_project.scripts.dataset_builder import build_dataset
from AT_AFO_project.scripts.model_trainer import train_model

def main():
    X, y = build_dataset()
    print(f"총 데이터 수: {len(X)}")
    model, scaler = train_model(X, y)

if __name__ == "__main__":
    main()
