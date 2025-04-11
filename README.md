
# Hybrid Models in Quantitative Finance

This repository provides an implementation of various hybrid models used in quantitative finance. The models featured include:  
- **BSHW (Black-Scholes Hull-White) Hybrid Model**  
- **Heston Hull-White Hybrid Model**  
- **Schobel-Zhu Hull-White Hybrid Model**  
- **Diversification Product Hull-White Model**  

The project is developed as a [Streamlit](https://streamlit.io) web application to visually demonstrate model dynamics and pricing simulations in a user-friendly interface. This README covers the theoretical background, the underlying mathematical formulations in LaTeX, practical implementation details, and industry practices—a resource especially useful if you are preparing for quant interviews.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background: Block 1 – Quant Finance Theory](#theoretical-background-block-1--quant-finance-theory)
    - [Hybrid Models: Concept & Motivation](#hybrid-models-concept--motivation)
    - [Mathematical Formulation in LaTeX](#mathematical-formulation-in-latex)
3. [Practical Implementation: Block 2 – Model Implementation & Usage](#practical-implementation-block-2----model-implementation--usage)
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

Below are the key equations underlying each hybrid model, fully formatted with LaTeX:

#### 1. Black-Scholes Hull-White (BSHW) Model

The asset price \( S_t \) dynamics are given by:  
$$
dS_t = r_t S_t\, dt + \sigma S_t\, dW_t^{(S)}
$$  
where:  
- \( r_t \) is the short rate from the Hull-White model,  
- \( \sigma \) is the constant volatility,  
- \( dW_t^{(S)} \) represents the Brownian motion for the asset process.

The Hull-White model for the short rate \( r_t \) is:  
$$
dr_t = \left(\theta_t - a\, r_t\right)dt + \sigma_r\, dW_t^{(r)}
$$  
with:  
- \( a \) as the mean reversion rate,  
- \( \theta_t \) as the time-dependent drift,  
- \( \sigma_r \) as the volatility of the short rate,  
- \( dW_t^{(r)} \) as the Brownian motion for the interest rate.

The correlation between the two processes is given by:  
$$
dW_t^{(S)} \, dW_t^{(r)} = \rho \, dt
$$

#### 2. Heston Hull-White Hybrid Model

The stochastic volatility process \( v_t \) is modeled as:  
$$
dv_t = \kappa (\theta_v - v_t) dt + \xi \sqrt{v_t}\, dZ_t
$$  
where:  
- \( \kappa \) is the rate of mean reversion of \( v_t \),  
- \( \theta_v \) is the long-term mean of \( v_t \),  
- \( \xi \) controls the volatility of volatility,  
- \( dZ_t \) represents a Brownian motion.

The asset price process under the Heston model combined with the Hull-White short rate is:  
$$
dS_t = r_t S_t\, dt + \sqrt{v_t} S_t\, dW_t^{(S)}
$$

#### 3. Schobel-Zhu Hull-White Hybrid Model

The Schobel-Zhu model for stochastic volatility is written as:  
$$
dv_t = \alpha (m - v_t)dt + \beta\, dZ_t
$$  
and the asset price process becomes:  
$$
dS_t = r_t S_t\, dt + f(v_t) S_t\, dW_t^{(S)}
$$  
where \( f(v_t) \) represents a function capturing the dependency on the volatility process.

#### 4. Diversification Product Hull-White Model

This model integrates diversification of risk factors with the Hull-White short rate. A general formulation can be expressed as:  
$$
dr_t = \left(\theta_t - a\, r_t\right)dt + \sigma_r\, dW_t^{(r)} + \sum_{i} \lambda_i X_t^{(i)}
$$  
where:  
- \( X_t^{(i)} \) represents additional risk factors,  
- \( \lambda_i \) denotes the sensitivity to each factor.


## Practical Implementation: Block 2 – Model Implementation & Usage

### Repository Structure & Running the App

The repository is organized as follows:

```
├── README.md               # Project overview and documentation
├── requirements.txt        # List of Python package dependencies
├── streamlit_app.py        # Main Streamlit application script
├── models/                 # Directory containing model implementations
│   ├── bshw.py             # Implementation of the BSHW hybrid model
│   ├── heston_hw.py        # Implementation of the Heston Hull-White model
│   ├── schobel_zhu.py      # Implementation of the Schobel-Zhu Hull-White model
│   └── diversification.py  # Implementation of the Diversification Product Hull White model
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
   Open your browser and navigate to the URL provided in the terminal (typically, `http://localhost:8501`).

### Calibration, Simulation, and Industry Practice Tips

- **Model Calibration:**  
  Calibrate parameters using market data. For example, align the yield curve and the implied volatility surface by minimizing the pricing error between the model and market prices.
  
- **Simulation Techniques:**  
  Use Monte Carlo simulation to generate sample paths that respect the correlation between asset prices and interest rates.
  
- **Practical Example:**  
  In the Heston Hull-White model, you might calibrate using:
  - Market-implied volatility \( (\sigma) \) for the asset,
  - Historical short rate data for the Hull-White process,
  - And optimization routines (e.g., least squares) to adjust parameters \( a \), \( \sigma_r \), \( \kappa \), \( \theta_v \), and \( \xi \).

- **Industry Practice Tips:**  
  - **Robust Parameter Estimation:** Ensure thorough data cleaning and robust optimization—regularization can be very beneficial.
  - **Risk Sensitivity Analysis:** Analyze how option prices change with respect to key model parameters.
  - **Backtesting:** Evaluate model performance using historical data to gauge robustness across various market conditions.
  
> **Nitish Sir Hinglish Note:**  
> "Yaar, jab interview ke liye prepare kar rahe ho, to practice karna mat bhoolna. Code ko samajh lo, data se interact karo, aur real market ke scenarios pe apne model chalake dekho. Thoda risk sensitivity analysis kar lo, tabhi industry mein entry milegi!"

---

## Examples and Use Cases

- **Option Pricing:**  
  Hybrid models can price exotic options where both interest rate uncertainty and volatility dynamics significantly impact the pricing.
  
- **Risk Management:**  
  Generate market scenarios to forecast Value at Risk (VaR) or Expected Shortfall (ES) when multiple risk factors are at play.
  
- **Portfolio Optimization:**  
  Dynamic hedging strategies can be optimized by modeling asset prices, volatilities, and interest rates jointly.

---

## Additional Tips for Quant Interview Preparation

- **Understand the Underlying Mathematics:**  
  Be proficient in deriving and explaining the stochastic differential equations (SDEs) that constitute each hybrid model.
- **Practice Model Calibration:**  
  Build calibration routines and understand error minimization in the context of market data.
- **Review Industry Applications:**  
  Know how these models are used in risk management and derivative pricing within financial institutions.
- **Project Presentation:**  
  Prepare a concise walkthrough of your project detailing your implementation, calibration methods, and simulation techniques.

> **Nitish Sir Hinglish Note:**  
> "Interview mein bas theory nahi, coding aur practical implementation bhi clear hona chahiye. Ekdum step-by-step batao aur code demo dikhate hue apne analysis ko shaandar banao!"

---

## Conclusion

This repository serves as an all-in-one reference for hybrid models in quantitative finance. By integrating deep theoretical foundations with practical application techniques, it is an excellent tool for both academic exploration and industry practice. Use this resource to understand the mathematical elegance of hybrid models and to enhance your practical skills for real-world quant challenges.

---

## References

- Hull, J. C. *Options, Futures, and Other Derivatives*
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.*
- Schobel, R., & Zhu, J. (1999). *Stochastic Volatility with an Inhomogeneous Volatility of Volatility.*
- Additional research papers and industry white papers as needed.
```
