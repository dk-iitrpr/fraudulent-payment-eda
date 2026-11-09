# fraudulent-payment-eda

Predicting whether a payment transaction is fraudulent or legitimate using logistic regression.

<a href="#" style="background-color: #000000; color: #0055aa; padding: 4px 12px; border-radius: 20px; text-decoration: none; font-size: 12px; font-weight: bold; margin-right: 5px;">AI101 Mini-Project</a>
<a href="#" style="background-color: #000000; color: #0055aa; padding: 4px 12px; border-radius: 20px; text-decoration: none; font-size: 12px; font-weight: bold; margin-right: 5px;">Team 17</a>
<a href="#" style="background-color: #000000; color: #0055aa; padding: 4px 12px; border-radius: 20px; text-decoration: none; font-size: 12px; font-weight: bold;">Python 3.12+</a>

***TEAM and their Roles***
| Member | Entry no. | Role |
| :--- | :--- | :--- |
| Dikshant Kansal | 2026EEB1345 | Documentation lead |
| Anish Meena | 2026AIB1016 | Run and reproducibility lead |
| Nikita Patel | 2026ESB1484 | PPT lead |
| Ayati Jain | 2026MCB1520 | Video lead |

*All four members contributed code (see the commits)*

<span>
# Project Structure
fraudulent-payment-eda/</br>
├──data/dataset-part-001.csv</br>
|---data/dataset-part-002.csv</br>
|---data/dataset-part-003.csv</br>
|---data/dataset-part-004.csv</br>
|---data/dataset-part-005.csv#the original dataset contains 6362620 rows and 11 columns
</span>


digital-payment-fraud-detection/</br>
├── data/</br>
│&emsp;├── dataset-part-001.csv</br>
│&emsp;├── dataset-part-002.csv</br>
│&emsp;├── dataset-part-003.csv</br>
│&emsp;├── dataset-part-004.csv</br>
│&emsp;└── dataset-part-005.csv&emsp;&emsp;# The original dataset contains 6362620 rows and 11 columns
│&emsp;└── processed/            # Cleaned / feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                   # Saved trained models
├── reports/
│   └── figures/              # Plots: PR curve, confusion matrix, SHAP
├── requirements.txt
└── README.md