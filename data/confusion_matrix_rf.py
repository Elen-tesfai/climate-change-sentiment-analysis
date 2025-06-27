import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample true labels and predicted labels (replace with your actual test data)
y_true = ['positive', 'neutral', 'negative', 'positive', 'neutral', 'negative', 'positive', 'neutral']
y_pred = ['positive', 'neutral', 'neutral', 'positive', 'negative', 'negative', 'neutral', 'neutral']

# Define class names in order
classes = ['positive', 'neutral', 'negative']

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Random Forest Classifier')

# Save figure to file in your data folder
plt.savefig('confusion_matrix_rf.png')
plt.show()