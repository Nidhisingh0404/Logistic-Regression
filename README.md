 # Interview Question Answers

## Difference Between Precision and Recall

- **Precision**: The proportion of true positives out of all positive predictions.

  Precision = TP / (TP + FP)

- **Recall**: The proportion of true positives out of all actual positives.

  Recall = TP / (TP + FN)

---

## Cross-Validation in Binary Classification

- Cross-validation splits the dataset into multiple folds (e.g., k-folds).
- The model is trained on k-1 folds and validated on the remaining fold.
- This process is repeated k times with different validation folds.
- It helps to:
  - Reduce overfitting
  - Assess model generalization on unseen data
