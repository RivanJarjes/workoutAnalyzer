import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

# Read the merged CSV file
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "merged_training_data.csv"))
data = pd.read_csv(path)

# Display the columns to verify the structure
print("Columns in CSV:", data.columns.tolist())

# Define the feature columns
feature_columns = [
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_v",
    "right_elbow_x", "right_elbow_y", "right_elbow_z", "right_elbow_v",
    "right_wrist_x", "right_wrist_y", "right_wrist_z", "right_wrist_v",
    "elbow_angle", "head_angle", "spine_angle"
]

# Define label columns
label_columns = [
    "not_high", "not_low", "excessive_lean", "excessive_elbow_flare",
    "elbows_too_far"
]

# Analyze the data distribution
print("\n--- DATA ANALYSIS ---")
X = data[feature_columns].values
y = data[label_columns].values

for i, label in enumerate(label_columns):
    pos_count = np.sum(y[:, i] == 1)
    neg_count = np.sum(y[:, i] == 0)
    print(f"{label}: Positive={pos_count} ({pos_count/len(y)*100:.1f}%), Negative={neg_count} ({neg_count/len(y)*100:.1f}%)")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y[:, -1])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- TRAINING INDIVIDUAL MODELS ---")

# List to store individual models and their performance
models = []
results = []

# Train a separate model for each label
for i, label in enumerate(label_columns):
    print(f"\nTraining model for: {label}")
    
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
    log_reg.fit(X_train_scaled, y_train[:, i])
    lr_score = log_reg.score(X_test_scaled, y_test[:, i])
    print(f"Logistic Regression accuracy: {lr_score:.4f}")

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],), 
              kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', 
              kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(), Precision(), Recall()]
    )
    
    # Compute balanced class weights
    unique_classes = np.unique(y_train[:, i])
    if len(unique_classes) > 1:  # Only if we have both classes
        class_weight = compute_class_weight('balanced', classes=unique_classes, y=y_train[:, i])
        class_weight_dict = {c: w for c, w in zip(unique_classes, class_weight)}
    else:
        class_weight_dict = None
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    # Train with very small batch size
    history = model.fit(
        X_train_scaled, y_train[:, i],
        epochs=100,
        batch_size=8,
        validation_data=(X_test_scaled, y_test[:, i]),
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=0
    )
    
    # Evaluate
    eval_result = model.evaluate(X_test_scaled, y_test[:, i], verbose=0)
    print(f"Neural Network - Loss: {eval_result[0]:.4f}, Accuracy: {eval_result[1]:.4f}")
    
    # Compare with logistic regression and choose the better model
    if lr_score > eval_result[1]:
        print(f"Logistic Regression performs better for {label}")
        models.append(('logreg', log_reg, i))
        results.append(('logreg', label, lr_score))
    else:
        print(f"Neural Network performs better for {label}")
        models.append(('nn', model, i))
        results.append(('nn', label, eval_result[1]))
    
    # Plot training history for neural network
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss for {label}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy for {label}')
    plt.legend()
    plt.savefig(f"{label}_training.png")
    plt.close()

# Summarize results
print("\n--- FINAL RESULTS ---")
for model_type, label, accuracy in results:
    print(f"{label}: {model_type.upper()} - Accuracy: {accuracy:.4f}")

# Save models
for model_type, model_obj, idx in models:
    if model_type == 'nn':
        model_obj.save(f"{label_columns[idx]}_model.h5")
    elif model_type == 'logreg':
        with open(f"{label_columns[idx]}_logreg.pkl", 'wb') as f:
            pickle.dump(model_obj, f)

# Save the scaler for use in the main application
joblib.dump(scaler, 'feature_scaler.joblib')

