
from datapipeline.bike_data_pipeline import (
    run_training_pipeline,
    run_inference_pipeline
)

X_train, y_train, preprocessor = run_training_pipeline()
X_test = run_inference_pipeline(preprocessor)

print("Training features shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print("Test features shape:", X_test.shape)

