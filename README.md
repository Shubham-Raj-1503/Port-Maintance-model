<div align="center">

# ⚓ Port Equipment Predictive Maintenance

### 🔮 *Predicting Equipment Failures Before They Happen*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

*A smart Machine Learning solution that keeps ports running smoothly by predicting when equipment needs maintenance — before breakdowns occur.*

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🚢 The Problem

> **"An ounce of prevention is worth a pound of cure."**

Port equipment failures can cost **millions** in delays, damaged cargo, and emergency repairs. Traditional maintenance schedules are either:
- ⏰ **Time-based** — Often wasteful, replacing parts too early
- 🔧 **Reactive** — Too late, equipment already failed

## 💡 The Solution

This project uses **Random Forest Machine Learning** to analyze equipment behavior patterns and predict maintenance needs with precision. Think of it as giving your port equipment a **sixth sense** for detecting problems!

<div align="center">

```
📊 Data → 🧠 ML Model → ⚡ Predictions → 🛡️ Prevention
```

</div>

---

## ✨ Features at a Glance

| 🎯 Feature | 📝 Description |
|:---:|---|
| 🔮 | **Predictive Classification** — Binary prediction: *needs maintenance* or *good to go* |
| 📊 | **Feature Importance** — Discover what factors matter most |
| 📈 | **Rich Visualizations** — Beautiful charts that tell the story |
| ⚡ | **Fast Training** — Results in seconds, not hours |
| 🎨 | **Clean Code** — Well-documented, easy to understand |

---

## 🔬 How It Works

### 📡 Input Sensors

The model ingests **13 operational parameters** from port equipment:

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚡ power_consumption    │  🌡️ temperature      │  💧 humidity   │
├─────────────────────────────────────────────────────────────────┤
│  📅 equipment_age_days   │  ⏱️ operational_hours │  📦 load_%    │
├─────────────────────────────────────────────────────────────────┤
│  🔌 voltage_variation    │  ⚙️ power_factor      │  📳 vibration │
├─────────────────────────────────────────────────────────────────┤
│  🚢 ships_berthed        │  🕐 hour   │  📆 day   │  📅 month    │
└─────────────────────────────────────────────────────────────────┘
```

### 🚨 Maintenance Triggers

The system flags equipment when danger patterns emerge:

```python
🔴 ALERT CONDITIONS:
├── ⚡ High power (>2000 kWh) + 👴 Old equipment (>5 years)
├── ⚙️ Low power factor (<0.85)
├── 📳 High vibration (>2.5) + 📦 Heavy load (>90%)
├── 🔌 Voltage swing (>±8%)
├── 👴 Equipment age (>8 years)
└── ⏱️ Long operation (>20h) + 📦 Heavy load (>85%)
```

---

## 🎨 Visualizations

<div align="center">

| Chart | What It Shows |
|:-----:|:-------------|
| 📊 **Feature Importance** | Which factors drive maintenance decisions |
| 📈 **Age Distribution** | Equipment age spread across the port |
| 📦 **Power Boxplot** | Power consumption patterns & outliers |
| 🔵 **Scatter Plot** | Load vs Vibration with maintenance overlay |
| 📉 **Time Series** | Maintenance trends over time |

</div>

---

## 🚀 Quick Start

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Shubham-Raj-1503/Port-Maintance-model.git

# Navigate to project
cd Port-Maintance-model

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### ▶️ Run the Model

```bash
# Launch Jupyter
jupyter notebook port_maintenance_MLMODEL-.ipynb
```

Then just **Run All Cells** and watch the magic happen! ✨

---

## 🧠 Model Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                    🌲 RANDOM FOREST 🌲                       │
│                                                              │
│   ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐              │
│   │ 🌳 1 │  │ 🌳 2 │  │ 🌳 3 │  ...  │🌳 100│              │
│   └──┬───┘  └──┬───┘  └──┬───┘       └──┬───┘              │
│      │         │         │              │                   │
│      └─────────┴────┬────┴──────────────┘                   │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │  🗳️ VOTE   │                                │
│              └──────┬──────┘                                │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │ 🎯 PREDICT │                                 │
│              └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

</div>

| Parameter | Value |
|:---------:|:-----:|
| 🌲 Trees | 100 |
| 📐 Scaler | StandardScaler |
| 📊 Split | 80% Train / 20% Test |
| 🎲 Seed | 42 |

---

## 📁 Project Structure

```
Port-Maintance-model/
│
├── 📓 port_maintenance_MLMODEL-.ipynb   # 🧠 Main ML notebook
│
└── 📖 README.md                          # 📚 You are here!
```

---

## 🔮 Future Roadmap

<div align="center">

| Phase | Enhancement | Status |
|:-----:|:------------|:------:|
| 1️⃣ | Real-time sensor integration | 🔜 |
| 2️⃣ | REST API deployment | 🔜 |
| 3️⃣ | Model comparison (XGBoost, Neural Net) | 🔜 |
| 4️⃣ | Hyperparameter optimization | 🔜 |
| 5️⃣ | Cross-validation implementation | 🔜 |
| 6️⃣ | Dashboard visualization | 🔜 |

</div>

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

---

## 📜 License

This project is for **educational and research purposes**.

---

<div align="center">

### ⭐ Star this repo if you find it useful!

Made with ❤️ for smarter ports

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

**[⬆ Back to Top](#-port-equipment-predictive-maintenance)**

</div>
