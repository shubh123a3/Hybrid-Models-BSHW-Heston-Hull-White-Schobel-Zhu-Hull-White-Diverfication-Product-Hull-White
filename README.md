
# Hybrid Models: BSHW, Heston Hull-White, Schöbel-Zhu Hull-White, Diversification Product Hull-White

This project presents a unified framework to implement and analyze **Hybrid Financial Models** by combining **Stochastic Volatility Models** and **Stochastic Interest Rate Models**. The models included in this repository are:

- **Black-Scholes Hull-White (BSHW)**
- **Heston Hull-White (HHW)**
- **Schöbel-Zhu Hull-White (SZHW)**
- **Diversification Product Hull-White (DPHW)**

## 📌 Project Goal

To model and simulate financial derivatives by integrating interest rate dynamics (Hull-White model) with popular stochastic volatility models to more accurately reflect real-world market behavior. These models are widely used in the pricing of **long-dated options**, **hybrid derivatives**, and **structured products**.

---

## 🔧 Models Explained

### 1. Black-Scholes Hull-White (BSHW) Model

A hybrid extension of the Black-Scholes model that incorporates stochastic interest rates using the Hull-White process.

**Model Equations**:
- Asset dynamics:  
  $$ dS_t = (r_t - q) S_t dt + \sigma S_t dW^S_t $$
- Interest rate dynamics (Hull-White):  
  $$ dr_t = (\theta(t) - a r_t) dt + \eta dW^r_t $$

Where:
- \( S_t \): asset price  
- \( r_t \): short rate  
- \( \sigma \): asset volatility  
- \( q \): dividend yield  
- \( a, \eta \): Hull-White parameters  
- \( \theta(t) \): time-dependent drift to fit the initial term structure  
- \( W^S_t \), \( W^r_t \): correlated Brownian motions

---

### 2. Heston Hull-White (HHW) Model

Combines the **Heston stochastic volatility model** with the **Hull-White interest rate model**.

**Model Equations**:
- Asset price:  
  $$ dS_t = (r_t - q) S_t dt + \sqrt{v_t} S_t dW^S_t $$
- Volatility:  
  $$ dv_t = \kappa (\bar{v} - v_t) dt + \sigma_v \sqrt{v_t} dW^v_t $$
- Interest rate:  
  $$ dr_t = (\theta(t) - a r_t) dt + \eta dW^r_t $$

Correlation:
- \( dW^S_t, dW^v_t, dW^r_t \) are correlated with specified correlation matrix.

---

### 3. Schöbel-Zhu Hull-White (SZHW) Model

This model uses the **Ornstein-Uhlenbeck process** for volatility (mean-reverting Gaussian process), suitable for modeling equity-linked products.

**Model Equations**:
- Asset price:  
  $$ dS_t = (r_t - q) S_t dt + \sigma_t S_t dW^S_t $$
- Volatility:  
  $$ d\sigma_t = \kappa (\bar{\sigma} - \sigma_t) dt + \xi dW^\sigma_t $$
- Interest rate:  
  $$ dr_t = (\theta(t) - a r_t) dt + \eta dW^r_t $$

Here, volatility follows a normal distribution rather than the square-root process in Heston.

---

### 4. Diversification Product Hull-White (DPHW)

This is a practical hybrid model used for pricing **diversification-linked structured products**. It assumes:

- A correlation structure between underlying assets and interest rates.
- Flexibility in payoffs (basket options, quanto products, range accruals, etc.).

This model adapts the Hull-White short rate model to multiple underlyings with a diversified payoff structure.

---

## 📊 Key Features

- Simulates hybrid dynamics using correlated Brownian motions.
- Supports Monte Carlo simulation.
- Ready-to-use Python functions for pricing and path generation.
- Realistic modeling of long-dated and hybrid derivatives.

---

## 🧠 Connection to Quant Finance

These hybrid models are used extensively in:

- **Structured Product Pricing** (e.g., callable range accruals, CMS spread options)
- **Risk Management** in interest rate derivatives
- **Exotic Option Valuation** where both interest rates and volatilities are uncertain
- **Insurance Products** (e.g., Variable Annuities with Guaranteed Minimum Benefits)

---

## 🧪 Tools & Libraries

- Python (NumPy, SciPy, Pandas, Matplotlib)
- Jupyter Notebooks for simulations
- Scikit-learn (optional for regression-based pricing)
- QuantLib (integration planned)

---

## 📂 Folder Structure

```
Hybrid-Models/
│
├── BSHW_Model.py
├── Heston_HullWhite_Model.py
├── SZ_HullWhite_Model.py
├── Diversified_HullWhite_Model.py
├── utils.py
├── README.md
└── notebooks/
    ├── BSHW_Simulation.ipynb
    ├── HHW_Pricing.ipynb
    └── SZHW_Simulation.ipynb
```

---

## 📈 Future Improvements

- Add support for calibration to market data
- Integrate QuantLib for advanced pricing tools
- Portfolio hedging using Greeks under hybrid dynamics
- GUI using Streamlit for interactive model input

---

## 🤝 Contribution

Contributions and feedback are welcome. Please open an issue or submit a pull request.

---

## 👤 Author

**Shubh Shrishrimal**  
Quantitative Finance Enthusiast | BSc CS | Machine Learning x Derivatives  
> Aim: To break into quant roles and build powerful financial tools.

---

## 🧠 Nitish Sir Style Hinglish Explanation (Concept Recap)

**"Yeh models tab kaam aate hai jab tumhare paas long-term derivatives hote hai jisme interest rate bhi move karta hai aur volatility bhi random hai. Sirf Black-Scholes nahi chalega bhai, isliye Heston + Hull-White ya SZ + Hull-White jaise hybrid models use karte hai. Market ke real behavior ko capture karne ke liye yeh combo mast hai. Structured products, insurance aur risk management mein inka full use hota hai!"**

---

## ⭐ Star this repo if you found it helpful!
```

