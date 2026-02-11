# Mobile Price Classification – ML Assignment 2 - Jyoti-2025aa05803

## Problem Statement
"The objective is to classify mobile phones into four price ranges (0–3) based on their specifications.  
This project demonstrates an end‑to‑end machine learning workflow: dataset preparation, model training, evaluation, and deployment with Streamlit."

---

## Dataset Description
- Dataset Name: Mobile Price Classification dataset 
- Source: Kaggle  
- Rows: ~2000 samples  
- Features: 20+ attributes (battery power, RAM, screen resolution, etc.)  
- Target: `price_range` (0 = low cost, 1 = medium cost, 2 = high cost, 3 = very high cost)

"The dataset is a good fit because it directly connects phone specifications to how much a phone costs.
Price ranges aren’t random, as they depend on the hardware and features like (battery power, RAM, screen resolution, etc.) inside the phone. 
This dataset includes those details, so it makes sense to use them for prediction."

---

## Models Implemented
Six classification models were trained and evaluated:
1. Logistic Regression  
2. Decision Tree  
3. K‑Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## Evaluation Metrics
| Model               | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|---------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression | 0.7575   | 0.9232 | 0.7561    | 0.7513 | 0.7531   | 0.6765 |
| Decision Tree       | 0.8325   | 0.8858 | 0.8294    | 0.8272 | 0.8267   | 0.7769 |
| kNN                 | 0.9425   | 0.9902 | 0.9404    | 0.9414 | 0.9408   | 0.9233 |
| Naive Bayes         | 0.7975   | 0.9560 | 0.7983    | 0.7926 | 0.7929   | 0.7313 |
| Random Forest       | 0.8925   | 0.9826 | 0.8916    | 0.8914 | 0.8905   | 0.8572 |
| XGBoost             | 0.9050   | 0.9913 | 0.9026    | 0.9046 | 0.9030   | 0.8735 |

---

## Observations
| Model               | Notes |
|---------------------|-------|
| Logistic Regression | Works as a basic model. It gives decent results but doesn’t capture complex patterns as well as tree‑based models |
| Decision Tree       | Strong accuracy but prone to overfitting. |
| kNN                 | Produced the best accuracy and MCC. It predicts well but can be slow when the dataset grows. |
| Naive Bayes         | Gave average results. Its assumption that features are independent doesn’t fully match this dataset, so performance is limited. |
| Random Forest       | Strong and reliable. By combining many trees, it avoids overfitting and gives stable results. |
| XGBoost             | Balanced and powerful. It achieved high AUC and strong precision/recall, making it one of the best overall performers. |


---

