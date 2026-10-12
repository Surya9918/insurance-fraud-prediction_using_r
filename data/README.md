# Dataset Acquisition & Specification Guide

## Auto Insurance Fraud Dataset

### 1. Dataset Overview & Provenance
This project investigates auto insurance claims to detect fraudulent activities using supervised machine learning in R.

* **Primary Source (Kaggle):** [Auto Insurance Fraud Dataset by brijbhushannanda1979](https://www.kaggle.com/datasets/brijbhushannanda1979/auto-insurance-fraud-dataset)
* **Secondary Academic Source (Mendeley Data):** Aqqad, Abdelrahim (2023), *“insurance_claims”*, Mendeley Data, V2, [doi: 10.17632/992mh7dk9y.2](https://data.mendeley.com/datasets/992mh7dk9y/2)
* **Open Repository Mirror:** [insurancedata repository by mwitiderrick](https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv)
* **File Target Location:** `data/insurance_claims.csv`

---

### 2. Dataset Dimensions & Schema
* **Sample Size ($N$):** Exactly 1,000 policyholder claim records.
* **Feature Count:** 39 substantive predictor columns (an extra 40th empty column `_c39` may appear due to trailing delimiters in raw CSV exports).
* **Target Variable:** `fraud_reported`
  * Values in raw data: `"Y"` and `"N"` (or `"Yes"` and `"No"`).
  * Positive (Event) Class: `"Yes"` (Fraud).
  * Negative Class: `"No"` (Non-Fraud).
  * Class Distribution: 753 non-fraudulent claims (75.3%) vs. 247 fraudulent claims (24.7%).
  * Class Imbalance Ratio: ~3.05:1.

---

### 3. Data Dictionary

| # | Column Name | Type | Description / Domain |
|---|---|---|---|
| 1 | `months_as_customer` | Integer | Total months the customer has held an active policy |
| 2 | `age` | Integer | Age of the policyholder in years |
| 3 | `policy_number` | Integer | Unique identifier for the insurance policy (removed during preprocessing) |
| 4 | `policy_bind_date` | Date/String | Policy inception date (YYYY-MM-DD) |
| 5 | `policy_state` | Categorical | US State where policy is registered (`OH`, `IN`, `IL`) |
| 6 | `policy_csl` | Categorical | Combined Single Limits liability coverage (e.g., `250/500`, `100/300`, `500/1000`) |
| 7 | `policy_deductable` | Numeric | Policy deductible ($500, $1,000, $2,000) |
| 8 | `policy_annual_premium` | Numeric | Annual policy premium in USD |
| 9 | `umbrella_limit` | Numeric | Additional umbrella liability coverage amount |
| 10 | `insured_zip` | Integer | Policyholder residential ZIP code |
| 11 | `insured_sex` | Categorical | Gender of policyholder (`MALE`, `FEMALE`) |
| 12 | `insured_education_level` | Categorical | Education level (`MD`, `PhD`, `Associate`, `Masters`, `High School`, `College`, `JD`) |
| 13 | `insured_occupation` | Categorical | Policyholder occupation (14 categories) |
| 14 | `insured_hobbies` | Categorical | Policyholder hobbies/interests (20 categories) |
| 15 | `insured_relationship` | Categorical | Relationship of claimant (`hubby`, `own-child`, `not-in-family`, `other-relative`, `wife`, `unmarried`) |
| 16 | `capital-gains` | Numeric | Capital gains reported in financial profile |
| 17 | `capital-loss` | Numeric | Capital losses reported in financial profile |
| 18 | `incident_date` | Date/String | Date the vehicular incident occurred (YYYY-MM-DD) |
| 19 | `incident_type` | Categorical | Type of incident (`Single Vehicle Collision`, `Vehicle Theft`, `Multi-vehicle Collision`, `Parked Car`) |
| 20 | `collision_type` | Categorical | Specific collision mechanism (`Side Collision`, `Rear Collision`, `Front Collision`, or `?` if unknown) |
| 21 | `incident_severity` | Categorical | Severity level (`Minor Damage`, `Total Loss`, `Major Damage`, `Trivial Damage`) |
| 22 | `authorities_contacted` | Categorical | First responders contacted (`Police`, `Fire`, `Ambulance`, `Other`, `None`) |
| 23 | `incident_state` | Categorical | US State where incident transpired (`NY`, `SC`, `WV`, `VA`, `NC`, `PA`, `OH`) |
| 24 | `incident_city` | Categorical | City where incident occurred (`Springfield`, `Arlington`, `Columbus`, `Northbend`, `Hillsdale`, `Riverwood`, `Northbrook`) |
| 25 | `incident_location` | String | Street address / intersection where incident took place |
| 26 | `incident_hour_of_the_day`| Integer | Hour of collision in 24-hour format (0–23) |
| 27 | `number_of_vehicles_involved` | Integer | Total vehicles involved in the collision |
| 28 | `property_damage` | Categorical | Whether property damage occurred (`YES`, `NO`, or `?`) |
| 29 | `bodily_injuries` | Integer | Count of individuals suffering bodily injuries (0, 1, 2) |
| 30 | `witnesses` | Integer | Number of recorded witnesses (0, 1, 2, 3) |
| 31 | `police_report_available` | Categorical | Availability of official police report (`YES`, `NO`, or `?`) |
| 32 | `total_claim_amount` | Numeric | Total claim compensation demanded in USD |
| 33 | `injury_claim` | Numeric | Portion of claim allocated to bodily injury medical expenses |
| 34 | `property_claim` | Numeric | Portion of claim allocated to non-vehicular property damages |
| 35 | `vehicle_claim` | Numeric | Portion of claim allocated to vehicular repair/replacement |
| 36 | `auto_make` | Categorical | Vehicle manufacturer (14 makes: `Saab`, `Dodge`, `Subaru`, `Nissan`, `Chevrolet`, `Ford`, `BMW`, `Toyota`, `Audi`, `Volkswagen`, `Accura`, `Mercedes`, `Jeep`, `Honda`) |
| 37 | `auto_model` | Categorical | Vehicle model (39 models) |
| 38 | `auto_year` | Integer | Vehicle manufacturing model year (1995–2015) |
| 39 | `fraud_reported` | Categorical | **Target variable**: Ground-truth fraud status (`Y`/`N` or `Yes`/`No`) |
| 40 | `_c39` | Empty | Artifact trailing comma column present in certain raw exports (dropped immediately) |

---

### 4. Critical Data Hygiene: The `?` Missing Value Convention
In this raw dataset, missing observations are encoded as the literal string character `?` rather than empty strings or `NA`.
Specifically:
* `collision_type`: Contains ~178 missing observations (`?`).
* `property_damage`: Contains ~360 missing observations (`?`).
* `police_report_available`: Contains ~343 missing observations (`?`).

The `01_load_data.R` and `03_preprocessing.R` scripts explicitly convert literal `?` values to R's native `NA` representation upon read:
```r
claims_raw <- readr::read_csv(
  "data/insurance_claims.csv",
  na = c("", "NA", "?")
)
```

---

### 5. Step-by-Step Acquisition Instructions

#### Method 1: Kaggle CLI (Recommended)
Ensure your Kaggle API credentials are configured in `~/.kaggle/kaggle.json`:
```bash
kaggle datasets download -d brijbhushannanda1979/auto-insurance-fraud-dataset -p data/ --unzip
```

#### Method 2: Manual Web Download
1. Navigate to [Kaggle Auto Insurance Fraud Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/auto-insurance-fraud-dataset).
2. Click **Download** (approx. 267 KB).
3. Extract `insurance_claims.csv` into the `data/` directory of this repository:
   ```bash
   cp /path/to/downloaded/insurance_claims.csv data/insurance_claims.csv
   ```

#### Method 3: Mendeley Data Direct Access
Download directly from [Mendeley Data DOI: 10.17632/992mh7dk9y.2](https://data.mendeley.com/datasets/992mh7dk9y/2) and place the resulting `insurance_claims.csv` into `data/`.

---

### 6. Academic Integrity Notice
In adherence to strict academic honesty standards:
* No synthetic or falsified data is committed in place of the empirical dataset.
* Kaggle API credentials and personal tokens are strictly excluded via `.gitignore`.
* The preprocessing pipeline is validated to behave identically on the official Kaggle dataset.
