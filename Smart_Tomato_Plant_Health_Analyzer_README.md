# 🌱 Smart Tomato Plant Health Analyzer

### AI-Powered Multi-View Disease Detection, Severity Estimation & Need-Based Spraying Decision Support

> **Hackathon Prototype \| Smart Agriculture \| Computer Vision + Deep
> Learning**

------------------------------------------------------------------------

## 📌 Project Overview

**Smart Tomato Plant Health Analyzer** is an AI-based decision-support
prototype designed to reduce unnecessary pesticide application in tomato
farming.

Traditional spraying often treats an entire field uniformly, even though
individual plants may have very different health conditions. This
project follows a **plant-level, need-based approach**:

**Capture 4 views of the same plant → Detect disease → Estimate
confidence → Estimate visible affected area → Estimate severity → Check
multi-view agreement → Generate treatment priority → Convert severity
into a prototype spray-duration command.**

The current prototype focuses on the **AI/software decision layer**.
Hardware actuation is designed as the next integration layer.

------------------------------------------------------------------------

# 🎯 Problem Being Solved

Conventional pesticide spraying can create several problems:

-   Uniform spraying despite different plant health conditions
-   Chemical wastage
-   Higher operating cost
-   Unnecessary exposure of healthy plants to chemicals
-   Difficulty in manually inspecting large numbers of plants
-   Delayed identification of disease symptoms

### Our approach

Instead of asking:

> "Should the whole field be sprayed?"

the system asks:

> **"What is the health condition of this individual plant, and what
> treatment priority should it receive?"**

This creates the foundation for **precision / need-based spraying**.

------------------------------------------------------------------------

# 🧠 Core Idea

The prototype analyzes **four images of the same tomato plant from
different views**.

For every image, the AI produces:

1.  Predicted disease
2.  Prediction confidence
3.  Estimated visible affected area
4.  Severity category

The four predictions are then combined using a **multi-view decision
layer**.

### High-level flow

``` text
             📷 IMAGE 1
                  │
             📷 IMAGE 2
                  │
             📷 IMAGE 3
                  ├──────► AI ANALYSIS
             📷 IMAGE 4
                  │
                  ▼
       ┌────────────────────────┐
       │ MobileNetV3-Large      │
       │ Disease Classification │
       └────────────────────────┘
                  │
                  ▼
       Disease + Confidence
                  │
                  ▼
       Visible Affected Area
                  │
                  ▼
             Severity
        Low / Medium / High /
             Very High
                  │
                  ▼
        Multi-View Consensus
                  │
                  ▼
        AI Reliability Check
                  │
                  ▼
       Treatment Priority
                  │
                  ▼
      Prototype Spray Duration
```

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
┌─────────────────────────────────────────────────────────────┐
│                    TOMATO PLANT                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                     4 Camera Views
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
       View 1               View 2               View 3 ... View 4
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    Image Preprocessing
                               │
                               ▼
                  MobileNetV3-Large Model
                               │
                               ▼
              ┌────────────────────────────┐
              │ Disease Classification     │
              │ + Confidence Score         │
              └──────────────┬─────────────┘
                             │
                             ▼
                Affected Area Estimation
                             │
                             ▼
                       Severity Engine
                             │
                             ▼
                 Multi-View Fusion Engine
                             │
                             ▼
                    Reliability Decision
                             │
                ┌────────────┴────────────┐
                │                         │
             Reliable                 Uncertain
                │                         │
                ▼                         ▼
       Treatment Priority             RE-SCAN
                │
                ▼
       Spray Decision Layer
                │
                ▼
     Hardware Controller Interface
                │
                ▼
     Pump / Valve / Nozzle System
```

------------------------------------------------------------------------

# 🧩 AI Model

## MobileNetV3-Large

The disease-classification model used in the prototype is:

**MobileNetV3-Large**

It is a lightweight convolutional neural network architecture suitable
for computer-vision applications where computational efficiency matters.

For this project, the model was trained for **10 tomato classes**.

------------------------------------------------------------------------

# 🍅 Dataset

The project uses the **PlantVillage tomato subset** available in the
downloaded dataset.

The tomato classes used in the current training pipeline are:

  \#   Class
  ---- ---------------------------------------------
  1    Tomato_Bacterial_spot
  2    Tomato_Early_blight
  3    Tomato_Late_blight
  4    Tomato_Leaf_Mold
  5    Tomato_Septoria_leaf_spot
  6    Tomato_Spider_mites_Two_spotted_spider_mite
  7    Tomato\_\_Target_Spot
  8    Tomato\_\_Tomato_YellowLeaf\_\_Curl_Virus
  9    Tomato\_\_Tomato_mosaic_virus
  10   Tomato_healthy

### Dataset statistics used in the training run

**Total images:** 16,036

``` text
Training:   11,225
Validation:  2,405
Testing:    2,406
```

The class distribution was preserved according to the dataset split
generated by the project pipeline.

------------------------------------------------------------------------

# 📊 Model Performance

The reported results from the completed test evaluation are:

  Metric                           Result
  -------------------------- ------------
  Test Accuracy                **99.00%**
  Test Precision               **99.11%**
  Test Recall                  **98.80%**
  Test Macro F1                **98.94%**
  Best Validation Macro F1     **99.43%**

### Important note

These metrics are the results of the current **PlantVillage-based test
set**.

They should **not** be interpreted as guaranteed real-world field
accuracy. Field photographs can differ substantially from controlled
dataset images because of:

-   lighting
-   camera quality
-   leaf orientation
-   background clutter
-   multiple leaves
-   occlusion
-   disease stage
-   weather
-   cultivar differences
-   camera distance

Therefore, the prototype should be presented as a **decision-support
system**, not as a certified autonomous pesticide prescription system.

------------------------------------------------------------------------

# 📈 Classification Report

The completed test evaluation produced the following class-level
results:

  ------------------------------------------------------------------------------------------------------
  Disease Class                                          Precision             Recall                 F1
  --------------------------------------------- ------------------ ------------------ ------------------
  Tomato_Bacterial_spot                                     99.38%            100.00%             99.69%

  Tomato_Early_blight                                      100.00%             93.33%             96.55%

  Tomato_Late_blight                                        99.30%             99.30%             99.30%

  Tomato_Leaf_Mold                                         100.00%             98.60%             99.30%

  Tomato_Septoria_leaf_spot                                 98.89%            100.00%             99.44%

  Tomato_Spider_mites_Two_spotted_spider_mite               95.82%             99.60%             97.67%

  Tomato\_\_Target_Spot                                     98.10%             98.57%             98.34%

  Tomato\_\_Tomato_YellowLeaf\_\_Curl_Virus                 99.58%             98.96%             99.27%

  Tomato\_\_Tomato_mosaic_virus                            100.00%            100.00%            100.00%

  Tomato_healthy                                           100.00%             99.58%             99.79%
  ------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 🔬 What Happens During Inference?

## Step 1 --- Four Images

The user uploads/captures exactly four views of the same plant.

Example:

``` text
                 SAME TOMATO PLANT

          ┌─────────┐
          │  View 1 │
          └────┬────┘
               │
          ┌────▼────┐
          │  View 2 │
          └────┬────┘
               │
          ┌────▼────┐
          │  View 3 │
          └────┬────┘
               │
          ┌────▼────┐
          │  View 4 │
          └─────────┘
```

Different views help the system observe different portions of the plant.

------------------------------------------------------------------------

# 🧠 Step 2 --- Disease Classification

Each image is independently passed through MobileNetV3-Large.

Example output:

``` text
View 1
Disease: Bacterial Spot
Confidence: 57.30%

View 2
Disease: Bacterial Spot
Confidence: 82.00%

View 3
Disease: Early Blight
Confidence: 77.45%

View 4
Disease: Early Blight
Confidence: 82.23%
```

The system then checks how consistently the four views agree.

------------------------------------------------------------------------

# 🎯 Step 3 --- Multi-View Consensus

The current prototype uses the four image predictions to determine the
most frequent disease.

Example:

``` text
Bacterial Spot → 2/4 views
Early Blight   → 2/4 views
```

This is treated as an **ambiguous / low-reliability situation**, rather
than blindly making a strong treatment decision.

This is important because a real plant may show different symptoms
across different leaves, and inconsistent predictions should trigger
caution.

------------------------------------------------------------------------

# 📐 Step 4 --- Affected Area Estimation

The prototype estimates the visible diseased/affected portion of the
image.

Example:

``` text
Leaf area pixels       : 13,432
Affected pixels        : 159
Affected area          : 1.18%
Severity               : Healthy / Very Low
```

For another image:

``` text
Affected area: 82.34%
Severity: Very High
```

### Important technical limitation

The current affected-area module is a **computer-vision approximation**.

It is not a ground-truth pixel-level disease segmentation model.

Therefore:

> **Affected Area ≠ medically/agronomically certified disease
> percentage.**

It should be described in the hackathon as an **estimated visible
affected area**.

------------------------------------------------------------------------

# 🌿 Step 5 --- Severity Engine

The estimated affected area is mapped to a severity category.

Conceptually:

``` text
Low affected area
        │
        ▼
      LOW
        │
        ▼
   MEDIUM
        │
        ▼
      HIGH
        │
        ▼
   VERY HIGH
```

The severity engine is used to convert image-level observations into a
simple operational decision.

------------------------------------------------------------------------

# 🤖 Step 6 --- AI Reliability

The system does not only look at disease confidence.

It also checks **agreement between the four views**.

Example:

``` text
4/4 same disease
→ High reliability

3/4 same disease
→ Moderate/High reliability

2/4 same disease
→ Low/Medium reliability depending on confidence

Conflicting predictions
→ Re-scan recommended
```

This prevents a single high-confidence image from automatically
controlling the whole plant-level decision.

------------------------------------------------------------------------

# 🚨 Step 7 --- Treatment Priority

The prototype produces a high-level treatment priority:

``` text
Healthy / Very Low
        ↓
   NO / MINIMAL ACTION

Low
        ↓
   LOW PRIORITY

Medium
        ↓
   MODERATE PRIORITY

High
        ↓
   HIGH PRIORITY

Very High
        ↓
   VERY HIGH PRIORITY
```

If AI reliability is low, the system can override the treatment decision
and show:

``` text
⚠️ RE-SCAN REQUIRED
```

This is a key safety feature of the prototype.

------------------------------------------------------------------------

# 💧 Step 8 --- Prototype Spray Decision Layer

For the hackathon demonstration, severity can be mapped to a **prototype
spray-duration command**:

  Severity               Prototype Command
  -------------------- -------------------
  Healthy / Very Low                 0 sec
  Low                                1 sec
  Medium                             3 sec
  High                               4 sec
  Very High                          5 sec
  Low AI reliability               Re-scan

### ⚠️ Important

These seconds are **prototype actuator-control values**, not real
pesticide dosage recommendations.

Actual pesticide dosage must be determined from the specific pesticide
label, crop stage, nozzle flow rate, pressure, calibration, field
conditions, and agricultural expert guidance.

The AI model itself does **not** determine a legally or agronomically
validated chemical dose.

------------------------------------------------------------------------

# 🦾 Hardware Architecture

The software architecture is designed so the future hardware system can
receive a command from the AI decision layer.

``` text
                  📷 CAMERA
                     │
                     ▼
             Edge Computer
        (Raspberry Pi / Jetson etc.)
                     │
                     ▼
            AI Inference Engine
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Disease      Affected       Confidence
   Detection        Area
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Severity Engine
                     │
                     ▼
             Reliability Check
                     │
          ┌──────────┴──────────┐
          │                     │
        LOW                   HIGH
      RELIABILITY            RELIABILITY
          │                     │
          ▼                     ▼
       RE-SCAN            Spray Decision
                                │
                                ▼
                         Microcontroller
                         (Arduino/ESP32)
                                │
                                ▼
                         Relay / MOSFET
                                │
                                ▼
                           Pump / Valve
                                │
                                ▼
                              Nozzle
                                │
                                ▼
                     🎯 TARGET PLANT ONLY
```

------------------------------------------------------------------------

# 🔌 Future Hardware Components

A possible physical prototype can contain:

### Vision

-   Camera module / USB camera
-   Fixed camera mount
-   Four-view capture arrangement

### Processing

-   Raspberry Pi / NVIDIA Jetson / similar edge computer
-   Or laptop during prototype stage

### Controller

-   ESP32 / Arduino

### Actuation

-   DC pump
-   Solenoid valve
-   Relay or MOSFET driver
-   Spray nozzle
-   Tubing
-   Liquid reservoir

### Optional

-   Flow sensor
-   Pressure sensor
-   Distance sensor
-   Plant-position sensor
-   Emergency stop
-   Manual override

------------------------------------------------------------------------

# 🧠 Complete Decision Logic

``` text
START
  │
  ▼
Capture 4 images
  │
  ▼
Are all 4 images valid?
  │
  ├── NO ─────► Request another image
  │
  └── YES
        │
        ▼
Run disease classification
        │
        ▼
Estimate confidence
        │
        ▼
Estimate visible affected area
        │
        ▼
Estimate severity
        │
        ▼
Fuse four views
        │
        ▼
Check model agreement
        │
        ├── LOW RELIABILITY
        │        │
        │        ▼
        │   RE-SCAN REQUIRED
        │
        └── ACCEPTABLE RELIABILITY
                 │
                 ▼
          Determine severity
                 │
                 ├── Healthy/Very Low → 0 sec prototype command
                 ├── Low             → 1 sec
                 ├── Medium          → 3 sec
                 ├── High            → 4 sec
                 └── Very High       → 5 sec
                                      │
                                      ▼
                              Hardware Controller
                                      │
                                      ▼
                               Pump / Valve
                                      │
                                      ▼
                               Targeted Spray
                                      │
                                      ▼
                                     END
```

------------------------------------------------------------------------

# 🖥️ Streamlit Application

The current prototype includes a Streamlit-based user interface.

## Main UI Workflow

``` text
┌──────────────────────────────────────────────┐
│ 🌱 Smart Tomato Plant Health Analyzer       │
│                                              │
│ AI-powered 4-image plant scanning            │
│                                              │
│ 📷 Upload 4 Images                           │
│ [Image 1] [Image 2] [Image 3] [Image 4]     │
│                                              │
│             🔍 ANALYZE PLANT                 │
└──────────────────────────────────────────────┘
                         │
                         ▼
             Final Plant Health Report
```

The UI displays:

-   Disease
-   Confidence
-   Consensus
-   Affected area
-   Severity
-   AI reliability
-   Recommended action
-   Individual image analysis
-   Disease consensus
-   Decision explanation

------------------------------------------------------------------------

# 📸 Suggested README Screenshots

Place your screenshots inside:

``` text
docs/
└── screenshots/
    ├── 01_upload_interface.png
    ├── 02_final_health_report.png
    ├── 03_ai_decision_explanation.png
    ├── 04_individual_image_analysis.png
    ├── 05_disease_consensus.png
    └── 06_hardware_architecture.png
```

Then add them to this README using:

``` markdown
## 📱 Application Interface

![Upload Interface](docs/screenshots/01_upload_interface.png)

![Final Health Report](docs/screenshots/02_final_health_report.png)

![AI Decision Explanation](docs/screenshots/03_ai_decision_explanation.png)

![Individual Image Analysis](docs/screenshots/04_individual_image_analysis.png)

![Disease Consensus](docs/screenshots/05_disease_consensus.png)
```

### Recommended screenshots from your current prototype

**01_upload_interface.png**

Use the screenshot showing:

> "Smart Tomato Plant Health Analyzer"\
> "Upload 4 Images"\
> Four uploaded tomato images.

**02_final_health_report.png**

Use the screenshot showing:

> "Final Plant Health Report"\
> Disease / Confidence / Consensus / Affected Area\
> Severity / AI Reliability / Recommended Action

**03_ai_decision_explanation.png**

Use the screenshot showing:

> "AI Decision Explanation"\
> Most frequent prediction\
> Fused model confidence\
> Robust estimated affected area\
> Estimated severity\
> AI reliability

**04_individual_image_analysis.png**

Use the screenshot showing:

> View 1 / View 2 / View 3 / View 4\
> Disease / Confidence / Affected Area

**05_disease_consensus.png**

Use the section showing:

> Disease Consensus\
> Bacterial Spot → 2/4 images\
> Early Blight → 2/4 images

------------------------------------------------------------------------

# 🧪 Example End-to-End Result

Example:

``` text
4 images captured

View 1
Disease: Bacterial Spot
Confidence: 57.30%
Affected Area: 44.16%

View 2
Disease: Bacterial Spot
Confidence: 82.00%

View 3
Disease: Early Blight
Confidence: 77.45%
Affected Area: 81.02%

View 4
Disease: Early Blight
Confidence: 82.23%
Affected Area: 82.34%
```

The system identifies:

``` text
Bacterial Spot → 2/4
Early Blight   → 2/4
```

Because the views disagree, the AI reliability becomes low.

Therefore:

``` text
Severity: Very High
AI Reliability: LOW
Recommended Action: RE-SCAN REQUIRED
```

This is preferable to automatically spraying a plant based on uncertain
evidence.

------------------------------------------------------------------------

# 💡 Why Multi-View Analysis?

A single image may not represent the complete condition of a plant.

For example:

``` text
              Plant
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
    Front      Side      Back
      │         │         │
      └─────────┼─────────┘
                ▼
        Better observation
```

Four views provide more visual information and allow the system to
identify disagreement.

This makes the prototype more suitable for a real-world scanning
workflow than a simple single-image classifier.

------------------------------------------------------------------------

# 🚀 Future Improvements

The current prototype can be upgraded significantly.

## 1. True Disease Segmentation

Current:

``` text
Affected area estimation
```

Future:

``` text
Original Image
      ↓
Segmentation Model
      ↓
Disease Mask
      ↓
Pixel-level affected area
```

Possible future models:

-   U-Net
-   DeepLabV3+
-   YOLO segmentation
-   SegFormer

------------------------------------------------------------------------

## 2. Better Real-World Dataset

PlantVillage images are useful for initial training, but field
deployment requires additional real-world images.

Future dataset:

``` text
PlantVillage
      +
Farm Images
      +
Different Lighting
      +
Different Cameras
      +
Different Growth Stages
      +
Different Cultivars
```

------------------------------------------------------------------------

## 3. Camera-Based Automated Scanning

Instead of manually uploading four images:

``` text
Plant enters scanning area
        ↓
Camera captures 4 views
        ↓
AI analyzes plant
        ↓
Decision generated
```

------------------------------------------------------------------------

## 4. Real-Time Edge AI

The model can eventually run directly on:

-   Raspberry Pi
-   NVIDIA Jetson
-   Edge AI device

This would reduce dependence on cloud inference.

------------------------------------------------------------------------

## 5. Hardware Integration

Future version:

``` text
AI Decision
     ↓
ESP32
     ↓
Pump Controller
     ↓
Solenoid Valve
     ↓
Nozzle
```

------------------------------------------------------------------------

## 6. Flow-Rate Based Spray Control

A stronger future system should not use only seconds.

Instead:

``` text
Required liquid volume
          ↓
Pump flow rate
          ↓
Pressure
          ↓
Nozzle characteristics
          ↓
Calculated valve-open duration
```

This is much more technically meaningful than treating seconds as a
universal pesticide dose.

------------------------------------------------------------------------

# 🛡️ Safety & Responsible AI

This project is a **hackathon prototype / decision-support system**.

It should not be presented as a certified pesticide-prescription system.

### The model should not independently decide:

-   Which chemical product to purchase
-   Chemical concentration
-   Legally permitted pesticide dose
-   Maximum residue limits
-   Application frequency
-   Human exposure safety
-   Environmental compliance

The prototype's spray-duration values are only **hardware demonstration
commands**.

Real deployment requires:

-   Agricultural expert validation
-   Pesticide-label compliance
-   Pump/nozzle calibration
-   Field testing
-   Safety interlocks
-   Manual override
-   Emergency stop
-   Environmental validation

------------------------------------------------------------------------

# 🏆 Innovation

The main innovation is not simply disease classification.

The project combines:

### 1️⃣ Disease Classification

``` text
What disease is visible?
```

### 2️⃣ Confidence

``` text
How confident is the model?
```

### 3️⃣ Affected Area

``` text
How much visible leaf area appears affected?
```

### 4️⃣ Severity

``` text
How serious is the estimated condition?
```

### 5️⃣ Multi-View Consensus

``` text
Do multiple views agree?
```

### 6️⃣ Reliability

``` text
Can the system safely make a decision?
```

### 7️⃣ Need-Based Action

``` text
What treatment priority should be assigned?
```

### 8️⃣ Hardware-Ready Interface

``` text
Can the decision be converted into an actuator command?
```

Together, these components transform a basic image classifier into a
**plant-level decision-support pipeline**.

------------------------------------------------------------------------

# 🧱 Current Prototype vs Future Production System

  ------------------------------------------------------------------------
  Component               Current Prototype       Future Version
  ----------------------- ----------------------- ------------------------
  Disease classification  ✅ MobileNetV3-Large    Improved/field-trained
                                                  model

  4-image analysis        ✅                      Automated multi-camera
                                                  capture

  Confidence              ✅                      Calibrated confidence

  Affected area           ✅ Approximation        True segmentation

  Severity                ✅ Rule-based           Validated severity model

  Consensus               ✅                      Advanced multi-view
                                                  fusion

  Reliability             ✅                      Calibrated uncertainty

  Streamlit UI            ✅                      Edge/mobile/web UI

  Spray command           ✅ Prototype duration   Flow-rate calibrated
                                                  control

  Hardware                🔜 Designed             Pump + valve + nozzle

  Field validation        🔜                      Required

  Expert validation       🔜                      Required
  ------------------------------------------------------------------------

------------------------------------------------------------------------

# 📂 Suggested Project Structure

``` text
smart-tomato-plant-health/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── tomato_mobilenetv3_best.pth
│
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── severity.py
│   ├── consensus.py
│   └── spray_decision.py
│
├── data/
│   └── README.md
│
├── docs/
│   ├── architecture.png
│   ├── hardware_architecture.png
│   └── screenshots/
│       ├── 01_upload_interface.png
│       ├── 02_final_health_report.png
│       ├── 03_ai_decision_explanation.png
│       ├── 04_individual_image_analysis.png
│       └── 05_disease_consensus.png
│
└── notebooks/
    └── smart_pesticide_sprayer.ipynb
```

------------------------------------------------------------------------

# ⚙️ Technology Stack

### AI / ML

-   Python
-   PyTorch
-   MobileNetV3-Large
-   Scikit-learn
-   NumPy
-   PIL / image processing

### Dataset

-   PlantVillage tomato dataset subset

### Application

-   Streamlit

### Development Environment

-   Google Colab
-   Google Drive

### Future Hardware

-   Camera
-   Raspberry Pi / Jetson
-   ESP32 / Arduino
-   Pump
-   Solenoid valve
-   Nozzle
-   Relay / MOSFET

------------------------------------------------------------------------

# 📌 Current Achievement

The current prototype successfully demonstrates:

``` text
✅ Tomato disease classification
✅ 10 disease/health classes
✅ MobileNetV3-Large
✅ 99.00% test accuracy
✅ 98.94% test Macro F1
✅ 99.43% best validation Macro F1
✅ 4-image multi-view analysis
✅ Confidence estimation
✅ Visible affected-area estimation
✅ Severity estimation
✅ Disease consensus
✅ AI reliability assessment
✅ Re-scan decision
✅ Streamlit interface
✅ Prototype spray-duration decision layer
✅ Hardware architecture for future integration
```

------------------------------------------------------------------------

# 🎤 Hackathon Pitch --- 30 Seconds

> **"Our system moves pesticide application from uniform field spraying
> to plant-level intelligent decision support. We capture four views of
> an individual tomato plant, use MobileNetV3-Large to detect disease,
> estimate confidence and visible affected area, calculate severity, and
> fuse the four views to measure decision reliability. If the evidence
> is reliable, the system generates a treatment priority and a prototype
> actuator command; if the views disagree, it asks for a re-scan instead
> of blindly spraying. The ultimate goal is targeted intervention,
> reduced chemical wastage, and more sustainable agriculture."**

------------------------------------------------------------------------

# 🧭 Roadmap

``` text
PHASE 1
Dataset + Disease Classification
             ✅

PHASE 2
Severity / Affected Area
             ✅

PHASE 3
4-Image Multi-View Fusion
             ✅

PHASE 4
Streamlit Decision Interface
             ✅

PHASE 5
Spray Decision Logic
             ✅ Prototype

PHASE 6
Hardware Integration
             🔜

PHASE 7
Flow Calibration
             🔜

PHASE 8
Field Dataset + Validation
             🔜

PHASE 9
Real-Time Edge Deployment
             🔜

PHASE 10
Production-Grade Precision Spraying
             🔜
```

------------------------------------------------------------------------

# 📜 Disclaimer

This repository describes a hackathon prototype for agricultural
decision support.

The reported AI metrics are based on the project's PlantVillage-derived
test split and should not be interpreted as field-level performance
guarantees.

Affected-area and severity values are computer-vision estimates.

The prototype spray-duration values are actuator demonstration
parameters and **must not be treated as pesticide dosage
recommendations**.

Any real agricultural deployment should be validated with agronomists,
calibrated hardware, approved pesticide labels, and appropriate
safety/regulatory procedures.

------------------------------------------------------------------------

# 🌱 Vision

### From:

``` text
Whole-field uniform spraying
```

### To:

``` text
Plant-level observation
        ↓
AI diagnosis
        ↓
Severity estimation
        ↓
Confidence + consensus
        ↓
Need-based intervention
        ↓
Targeted spraying
```

### Long-term goal

> **"Spray only where it is needed, only when the evidence is reliable,
> and only at a validated application level."**

------------------------------------------------------------------------

## ⭐ Project Status

**Hackathon Prototype --- AI Decision-Support Layer Completed**

**Current AI model:** MobileNetV3-Large\
**Classes:** 10 tomato classes\
**Test Accuracy:** 99.00%\
**Test Macro F1:** 98.94%\
**Best Validation Macro F1:** 99.43%\
**Multi-view:** 4 images\
**UI:** Streamlit\
**Hardware:** Architecture designed; physical integration pending
