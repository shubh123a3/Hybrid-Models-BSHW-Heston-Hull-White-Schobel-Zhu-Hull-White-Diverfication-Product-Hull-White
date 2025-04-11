

# Hybrid Models in Quantitative Finance

This repository provides an implementation of various hybrid models used in quantitative finance. The models featured include:  
- **BSHW (Black-Scholes Hull-White) Hybrid Model**  
- **Heston Hull-White Hybrid Model**  
- **Schobel-Zhu Hull-White Hybrid Model**  
- **Diversification Product Hull White Model**  

The project is developed as a [Streamlit](https://streamlit.io) web application to visually demonstrate model dynamics and pricing simulations in a user-friendly interface. This README covers the theoretical background, the underlying mathematical formulations in LaTeX, practical implementation details, and industry practices—a resource especially useful if you are preparing for quant interviews.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background: Block 1 – Quant Finance Theory](#theoretical-background-block-1-quant-finance-theory)
    - [Hybrid Models: Concept & Motivation](#hybrid-models-concept--motivation)
    - [Mathematical Formulation in LaTeX](#mathematical-formulation-in-latex)
3. [Practical Implementation: Block 2 – Model Implementation & Usage](#practical-implementation-block-2---model-implementation--usage)
    - [Repository Structure & Running the App](#repository-structure--running-the-app)
    - [Calibration, Simulation, and Industry Practice Tips](#calibration-simulation-and-industry-practice-tips)
4. [Examples and Use Cases](#examples-and-use-cases)
5. [Additional Tips for Quant Interview Preparation](#additional-tips-for-quant-interview-preparation)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Overview

This repository demonstrates advanced hybrid models that integrate different components used in modern quantitative finance. Hybrid models combine aspects of stochastic interest rates and stochastic volatility (or other dynamic factors) to more accurately capture market behavior. They are particularly useful in pricing complex derivatives and structured products. The app allows users to explore the simulation of these models, adjust key parameters, and visualize results in real time.

---

## Theoretical Background: Block 1 – Quant Finance Theory

### Hybrid Models: Concept & Motivation

Hybrid models in finance are designed to capture multiple sources of randomness simultaneously. For example:  
- **Interest Rate Component:** Typically modeled using the Hull-White process to represent the evolution of interest rates.
- **Asset Price or Volatility Component:** Modeled using either the Black-Scholes, Heston, or Schobel-Zhu dynamics to capture the stochastic behavior of asset prices or volatilities.

In a hybrid model, these processes are combined—allowing for correlations between interest rate movements and asset price fluctuations. This is particularly useful for pricing instruments where both rates and volatility influence the option value.

### Mathematical Formulation in LaTeX

Below are the key equations underlying each hybrid model:

#### 1. Black-Scholes Hull-White (BSHW) Model

The classical Black-Scholes dynamics for the asset price, \(S_t\), is given by:  
$$
dS_t = r_t S_t\, dt + \sigma S_t\, dW_t^{(S)}
$$  
where:
- \(r_t\) is the short rate from the Hull-White model,
- \(\sigma\) is the constant volatility,
- \(W_t^{(S)}\) is a Brownian motion for the asset process.

The Hull-White model for the short rate, \(r_t\), is:  
$$
dr_t = \left(\theta_t - a\, r_t\right)dt + \sigma_r\, dW_t^{(r)}
$$  
with:
- \(a\) as the mean reversion rate,
- \(\theta_t\) as the time-dependent drift,
- \(\sigma_r\) as the volatility of the short rate,
- \(W_t^{(r)}\) a Brownian motion for the interest rate.

These two stochastic differential equations (SDEs) are typically correlated, i.e.,  
$$
dW_t^{(S)} \, dW_t^{(r)} = \rho \, dt
$$

#### 2. Heston Hull-White Hybrid Model

The Heston model introduces a stochastic volatility process for the asset, \(v_t\):  
$$
dv_t = \kappa (\theta_v - v_t) dt + \xi \sqrt{v_t}\, dZ_t
$$  
where:
- \(\kappa\) is the rate at which \(v_t\) reverts to the mean \(\theta_v\),
- \(\xi\) controls the volatility of volatility,
- \(Z_t\) is another Brownian motion.
  
The asset price process in the Heston model combined with the Hull-White interest rate dynamic becomes:  
$$
dS_t = r_t S_t\, dt + \sqrt{v_t} S_t\, dW_t^{(S)}
$$

#### 3. Schobel-Zhu Hull-White Hybrid Model

The Schobel-Zhu model is another approach to model stochastic volatility. Its formulation can be summarized as:  
$$
dv_t = \alpha (m - v_t)dt + \beta\, dZ_t
$$  
with the asset price process:  
$$
dS_t = r_t S_t\, dt + f(v_t) S_t\, dW_t^{(S)}
$$  
where \(f(v_t)\) is a function capturing the effect of the underlying volatility process.

#### 4. Diversification Product Hull White Model

This model integrates diversification of risk factors with the Hull-White short rate. The diversification product approach may involve combining several independent short-rate models or mixing interest rate driven components with other risk factors, which can be generally written as:  
$$
dr_t = \left(\theta_t - a\, r_t\right)dt + \sigma_r\, dW_t^{(r)} + \sum_{i} \lambda_i X_t^{(i)}
$$  
where:
- \(X_t^{(i)}\) represent additional risk factors,
- \(\lambda_i\) denote the sensitivity to each factor.

> **Nitish Sir Hinglish Note:** "Bhai, ye hybrid models ekdum mast combination hain—jahan interest rate aur asset price dono ko ek saath capture kiya jata hai. Iska matlab, market ki real complexity ko samajhna thoda aasan ho jata hai."

---

## Practical Implementation: Block 2 – Model Implementation & Usage

### Repository Structure & Running the App

The repository is organized as follows:

```
├── README.md               # Project overview and documentation
├── requirements.txt        # List of Python package dependencies
├── streamlit_app.py        # Main Streamlit application script
├── models/                 # Directory containing model implementations
│   ├── bshw.py           # Implementation of the BSHW hybrid model
│   ├── heston_hw.py      # Implementation of the Heston Hull-White model
│   ├── schobel_zhu.py    # Implementation of the Schobel-Zhu Hull-White model
│   └── diversification.py # Implementation of the Diversification Product Hull White model
└── data/                   # Directory for sample data and simulation results
```

#### How to Run

1. **Install Dependencies:**  
   Create a virtual environment and install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Streamlit Application:**  
   Execute the main app script:  
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Explore the App:**  
   Open your browser and go to the URL provided in the terminal (typically, `http://localhost:8501`).

### Calibration, Simulation, and Industry Practice Tips

- **Model Calibration:**  
  Calibrate parameters using market data. For instance, match the yield curve and implied volatility surface by minimizing the error between model prices and observed market prices.
  
- **Simulation Techniques:**  
  Use Monte Carlo simulation to generate sample paths for both asset prices and interest rates. Ensure that correlations between the stochastic processes are respected.
  
- **Practical Example:**  
  In the Heston Hull-White model, calibrate using:
  - Market-implied volatility (\(\sigma\)) data for the asset,
  - Historical short rate data for the Hull-White process,
  - And use optimization routines (e.g., least squares) to adjust parameters \(a\), \(\sigma_r\), \(\kappa\), \(\theta_v\), and \(\xi\).

- **Industry Practice Tips:**  
  - **Robust Parameter Estimation:** Spend ample time on data cleaning and robust optimization. Use regularization methods if necessary.
  - **Risk Sensitivity Analysis:** Evaluate the models by analyzing the sensitivity of option prices to key parameters.
  - **Backtesting:** Always backtest your models using historical data to gauge performance over different market conditions.
  
> **Nitish Sir Hinglish Note:** "Yaar, jab interview ke liye prepare kar rahe ho, to practice karna mat bhoolna. Code ko samajh lo, data se interact karo, aur real market ke scenarios pe apna model chalake dekho. Thoda risk sensitivity analysis kar lo, tabhi industry mein entry milegi!"

---

## Examples and Use Cases

- **Option Pricing:**  
  Use the hybrid models to price exotic options where both interest rate uncertainty and volatility dynamics affect the price.
  
- **Risk Management:**  
  Simulate various market scenarios to forecast Value at Risk (VaR) or Expected Shortfall (ES), especially when interest rates and volatility might shift simultaneously.
  
- **Portfolio Optimization:**  
  Explore dynamic hedging strategies where asset prices, volatility, and interest rates are jointly modeled to optimize the risk-return profile.

---

## Additional Tips for Quant Interview Preparation

- **Understand the Underlying Mathematics:**  
  Ensure that you can derive and explain the SDEs for each component of the hybrid model.  
- **Practice Model Calibration:**  
  Build simple calibration routines and get comfortable with error analysis and optimization techniques.
- **Review Industry Applications:**  
  Discuss how these models help in managing risk and pricing derivatives in real-life financial institutions.
- **Project Presentation:**  
  Be ready to walk through your project implementation step-by-step. Highlight your choices in model selection, calibration methods, and simulation techniques.

> **Nitish Sir Hinglish Note:** "Interview mein bas theory nahi, coding aur practical implementation bhi clear hona chahiye. Ekdum step-by-step batao aur code demo dikhate hue apne analysis ko shaandar banao!"

---

## Conclusion

This repository serves as an all-in-one reference for hybrid models in quantitative finance. By integrating theoretical foundations with practical applications, it is an excellent tool for both academic exploration and industry practice. Use this resource not only to understand the mathematical elegance of hybrid models but also to gain insights into practical implementation challenges and solutions common in the quant finance industry.

---

## References

- Hull, J. C. *Options, Futures, and Other Derivatives*
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.
- Schobel, R., & Zhu, J. (1999). Stochastic Volatility with an Inhomogeneous Volatility of Volatility.
- Relevant research papers and industry white papers (attach or link as needed).

