from sklearn.metrics import classification_report,roc_curve, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

def evaluate(model, X_test, y_test, testing_dataset):
    pred = model.predict(X_test)
    pred_labels = np.argmax(pred, axis = 1)
    print("Labels: " ,pred_labels)
    loss, accuracy = model.evaluate(testing_dataset, batch_size=32)
    print("Loss is: ", loss)
    print("Accuracy is: ", accuracy)

    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    n_classes = pred.shape[1]
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves per Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


