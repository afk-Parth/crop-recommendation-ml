# crop-recommendation-ml
implementing logistic regression crop classification model using scikit learn
# Crop Recommendation 🌾🤖

A multi‑class classification model that suggests the most suitable crop to sow in a field based on soil and environmental conditions.

<table>
<tr><td>**Dataset**</td><td>Crop Recommendation Dataset — <a href="https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset">Kaggle</a></td></tr>
<tr><td>**Samples**</td><td>2 ,200 rows × 7 features</td></tr>
<tr><td>**Target**</td><td>22 crop labels (multinomial)</td></tr>
<tr><td>**Model**</td><td>Logistic Regression (multinomial)</td></tr>
</table>

---

## 1. Features
| Column | Description |
|--------|-------------|
| `N`, `P`, `K` | Soil macro‑nutrient values (kg / ha) |
| `temperature` | °C |
| `humidity` | % |
| `ph` | Soil pH |
| `rainfall` | mm |
| **Target → `label`** | Crop name (22 classes) |

---

## 2. Exploratory Data Analysis 
Key visualisations include:

* **Rainfall vs. crop** – box‑plots reveal rainfall requirements  
* **pH vs. crop** – violin‑plots show acidic / alkaline preferences  
* **Pair‑plot** – relationship between N, P, K & pH coloured by crop  
* **Class distribution** – count‑plot ensures no severe imbalance  

<p align="center">
  <em>(see notebook for charts)</em>
</p>

---

## 3. Pre‑processing
* Encoded crop names → `label_encoded` using `LabelEncoder`
* Train‑test split 80 / 20 with `random_state = 42`
* No missing values → dataset ready for modelling

---

## 4. Model & Training ⚙️
```python
model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500)
model.fit(X_train, y_train)
