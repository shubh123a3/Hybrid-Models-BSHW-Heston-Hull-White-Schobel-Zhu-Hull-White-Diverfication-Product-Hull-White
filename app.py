import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats
import enum
import scipy.optimize as optimize
import streamlit as st
import H1_HW_cos_mc
import SZHW_iV
import Diverfication_product

st.title("Hybrid Models : BSHW, Heston Hull White, Schobel-Zhu Hull White, Diverfication Product Hull White")

st.write("Please select a model from the sidebar.")
st.sidebar.title("Select a Model")
st.sidebar.write("Choose a model to visualize:")
st.subheader("Available models:")
st.write("1. BSHW (Black-Scholes-Hull-White)")
st.write("2.Heston Hull White (HestonHW)")
st.write("3.Schobel-Zhu Hull White (SZHW)")
st.write("4.Diverfication Product Hull White (DiverficationHW)")
st.sidebar.write("Select a model to visualize:")
def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# Sidebar for model selection
st.sidebar.title("Select a Model")
model = st.sidebar.selectbox("Model", ["Home", "BSHW", "HestonHW", "SZHW", "DiverficationHW"])

# Navigate to the selected page
if model == "BSHW":
    navigate_to_page("bshw")
elif model == "HestonHW":
    navigate_to_page("hestonhw")
elif model == "SZHW":
    navigate_to_page("szhw")
elif model == "DiverficationHW":
    navigate_to_page("diverficationhw")
else:
    navigate_to_page("home")

i = 1j

# time-step needed for differentiation
dt = 0.0001


# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
def CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, tau, K, N, L, P0T):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)
    # P0T  - zero-coupon bond for maturity T.

    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # assigning i=sqrt(-1)
    i = 1j
    x0 = np.log(S0 / K)

    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # sumation from k = 0 to k=N-1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a);

    # Determine coefficients for Put Prices
    H_k = CallPutCoefficients(OptionType.PUT, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(mat.dot(temp))

    # we use call-put parity for call options
    if CP == OptionType.CALL:
        value = value + S0 - K * P0T

    return value


# Determine coefficients for Put Prices
def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                       * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                         (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                           * np.pi * (c - a) / (
                                                                                                       b - a)) * np.exp(
        c)
    chi = chi * (expr1 + expr2)

    value = {"chi": chi, "psi": psi}
    return value


# Black-Scholes Call option price
def BS_Call_Option_Price(CP, S_0, K, sigma, tau, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0))
          * tau) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = stats.norm.cdf(d1) * S_0 - stats.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = stats.norm.cdf(-d2) * K * np.exp(-r * tau) - stats.norm.cdf(-d1) * S_0
    return value


# Implied volatility method
def ImpliedVolatilityBlack76(CP, marketPrice, K, T, S_0):
    func = lambda sigma: np.power(BS_Call_Option_Price(CP, S_0, K, sigma, T, 0.0) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, 0.2, tol=1e-9)
    # impliedVol = optimize.brent(func, brack= (0.05, 2))
    return impliedVol


def ChFBSHW(u, T, P0T, lambd, eta, rho, sigma):
    i = 1j
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)

    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)

    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) \
                      + eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    C = lambda u, tau: 1.0 / lambd * (i * u - 1.0) * (1.0 - np.exp(-lambd * tau))

    # define a grid for the numerical integration of function theta
    zGrid = np.linspace(0.0, T, 2500)
    term1 = lambda u: 0.5 * sigma * sigma * i * u * (i * u - 1.0) * T
    term2 = lambda u: i * u * rho * sigma * eta / lambd * (i * u - 1.0) * (T + 1.0 / lambd \
                                                                           * (np.exp(-lambd * T) - 1.0))
    term3 = lambda u: eta * eta / (4.0 * np.power(lambd, 3.0)) * np.power(i + u, 2.0) * \
                      (3.0 + np.exp(-2.0 * lambd * T) - 4.0 * np.exp(-lambd * T) - 2.0 * lambd * T)
    term4 = lambda u: lambd * np.trapz(theta(T - zGrid) * C(u, zGrid), zGrid)
    A = lambda u: term1(u) + term2(u) + term3(u) + term4(u)

    # Note that we don't include B(u)*x0 term as it is included in the COS method
    cf = lambda u: np.exp(A(u) + C(u, T) * r0)

    # Iterate over all u and collect the ChF, iteration is necessary due to the integration over in term4
    cfV = []
    for ui in u:
        cfV.append(cf(ui))

    return cfV


def BSHWVolatility(T, eta, sigma, rho, lambd):
    Br = lambda t, T: 1 / lambd * (np.exp(-lambd * (T - t)) - 1.0)
    sigmaF = lambda t: np.sqrt(sigma * sigma + eta * eta * Br(t, T) * Br(t, T) \
                               - 2.0 * rho * sigma * eta * Br(t, T))
    zGrid = np.linspace(0.0, T, 2500)
    sigmaC = np.sqrt(1 / T * np.trapz(sigmaF(zGrid) * sigmaF(zGrid), zGrid))
    return sigmaC


def BSHWOptionPrice(CP, S0, K, P0T, T, eta, sigma, rho, lambd):
    frwdS0 = S0 / P0T
    vol = BSHWVolatility(T, eta, sigma, rho, lambd)
    # As we deal with the forward prices we evaluate Black's 76 prices
    r = 0.0
    BlackPrice = BS_Call_Option_Price(CP, frwdS0, K, vol, T, r)
    return P0T * BlackPrice
# Render the selected page
if st.session_state["current_page"] == "bshw":
    st.title("BSHW Model")


    st.write("You selected BSHW (Black-Scholes-Hull-White) model.")
    st.write("This model is used for option pricing in a stochastic interest rate environment.")
    st.write("You can use this model to analyze the impact of stochastic interest rates on option prices.")
    action=st.radio("FInd option price or parmetric effect Implied volality ",["optionPrice","Impliedvolaity"])
   # if action==

    option_type = {"Call": OptionType.CALL, "Put": OptionType.PUT}
    slected_option = st.sidebar.selectbox("Select the option type", option_type.keys())
    CP = option_type[slected_option]

    K = np.linspace(40.0, 220.0, 100)
    K = np.array(K).reshape([len(K), 1])

    # HW model settings
    lambd = st.sidebar.number_input("Enter the mean reversion speed (lambda)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
    eta =st.sidebar.number_input("Enter the volatility of the short rate (eta)", value=0.05, min_value=0.01, max_value=1.0, step=0.01)
    sigma = st.sidebar.number_input("Enter the volatility of the stock (sigma)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)

    rho = st.sidebar.number_input("Enter the correlation between stock and interest rate (rho)", value=0.3, min_value=-1.0, max_value=1.0, step=0.01)

    S0 = st.sidebar.number_input("Enter the initial stock price (S0)", value=100.0, min_value=1.0, max_value=1000.0, step=1.0)

    T = st.sidebar.number_input("Enter the time to maturity (T)", value=5.0, min_value=0.01, max_value=10.0, step=0.01)
    if action=="optionPrice":

    # We define a ZCB curve (obtained from the market)
        P0T = lambda T: np.exp(-0.05 * T)

        N = 500
        L = 8

        # Characteristic function of the BSHW model + the COS method
        cf = lambda u: ChFBSHW(u, T, P0T, lambd, eta, rho, sigma)
        valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L, P0T(T))
        exactBSHW = BSHWOptionPrice(CP, S0, K, P0T(T), T, eta, sigma, rho, lambd)

        IV = np.zeros([len(K), 1])
        for idx in range(0, len(K)):
            frwdStock = S0 / P0T(T)
            valCOSFrwd = valCOS[idx] / P0T(T)
            IV[idx] = ImpliedVolatilityBlack76(CP, valCOSFrwd, K[idx], T, frwdStock)

        IVExact = BSHWVolatility(T, eta, sigma, rho, lambd)

        st.success( IVExact)
        # Plot option prices
        fig1 = plt.figure(1)
        plt.plot(K, valCOS)
        plt.plot(K, exactBSHW, '--r')
        plt.grid()
        plt.xlabel("strike")
        plt.ylabel("option price")
        plt.legend(["BSHW, COS method", "BSHW, exact solution"])
        plt.title("BSHW Option Prices")
        plt.tight_layout()
        st.pyplot(fig1)

        # Plot implied volatilities
        fig2 = plt.figure(2)
        plt.plot(K, IV * 100.0)
        plt.plot(K, np.ones([len(K), 1]) * IVExact * 100.0, '--r')
        plt.grid()
        plt.xlabel("strike")
        plt.ylabel("Implied Volatility [%]")
        plt.legend(["BSHW, COS method", "BSHW, exact solution"])
        plt.axis([np.min(K), np.max(K), 0, 100])
        plt.title("BSHW Implied Volatility")
        plt.tight_layout()
        st.pyplot(fig2)
    elif action=="Impliedvolaity":
        K = [100]
        K = np.array(K).reshape([len(K), 1])

        # We define a ZCB curve (obtained from the market)
        P0T = lambda T: np.exp(-0.05 * T)

        # Maturitires at which we compute implied volatility
        TMat = np.linspace(0.1, 5.0, 20)
        plot_placeholder = st.empty()
        co1, col2 = st.columns(2)
        with co1:
            paremetereffect = co1.selectbox("Select the parameter to vary", ["lambda", "eta","sigma", "rho"])
        if paremetereffect == "lambda":
            with col2:
                lambd_add = col2.number_input("Enter the value of lambda", min_value=0.0, max_value=1.0, value=0.05,

                                            step=0.01)
                lambdV = [0.001, 0.1, 0.5, 1.5]
                lambdV.append(lambd_add)

        if paremetereffect == "eta":
            with col2:
                eta_add = st.number_input("Enter the value of eta", min_value=0.0, max_value=1.0, value=0.05,
                                          step=0.01)

                etaV = [0.001, 0.05, 0.1, 0.15]
                etaV.append(eta_add)

        if paremetereffect == "sigma":
            with col2:
                sigma_add = st.number_input("Enter the value of sigma", min_value=0.0, max_value=1.0, value=0.05,
                                            step=0.01)

                sigmaV = [0.1, 0.2, 0.3, 0.4]
                sigmaV.append(sigma_add)

        if paremetereffect == "rho":
            with col2:
                rho_add = st.number_input("Enter the value of rho", min_value=-1.0, max_value=1.0, value=0.05,
                                          step=0.01)

                rhoV = [-0.7, -0.3, 0.3, 0.7]
                rhoV.append(rho_add)
        # Effect of lambda
        if paremetereffect == "lambda":
            plt.title("Effect of lambda on Implied Volatility")
            plot_placeholder = st.empty()
            fig1=plt.figure(1)
            plt.grid()
            plt.xlabel('maturity, T')
            plt.ylabel('implied volatility')
            legend = []
            for lambdaTemp in lambdV:
                IV = np.zeros([len(TMat), 1])
                for idx in range(0, len(TMat)):
                    T = TMat[idx]
                    val = BSHWOptionPrice(CP, S0, K, P0T(T), T, eta, sigma, rho, lambdaTemp)
                    frwdStock = S0 / P0T(T)
                    valFrwd = val / P0T(T)
                    IV[idx] = ImpliedVolatilityBlack76(CP, valFrwd, K, T, frwdStock)
                plt.plot(TMat, IV * 100.0)
                legend.append('lambda={0}'.format(lambdaTemp))
            plt.legend(legend)
            plot_placeholder.pyplot(fig1)
        # Effect of eta
        if paremetereffect == "eta":
            plt.title("Effect of eta on Implied Volatility")

            fig2=plt.figure(2)
            plt.grid()
            plt.xlabel('maturity, T')
            plt.ylabel('implied volatility')
            legend = []
            for etaTemp in etaV:
                IV = np.zeros([len(TMat), 1])
                for idx in range(0, len(TMat)):
                    T = TMat[idx]
                    val = BSHWOptionPrice(CP, S0, K, P0T(T), T, etaTemp, sigma, rho, lambd)
                    frwdStock = S0 / P0T(T)
                    valFrwd = val / P0T(T)
                    IV[idx] = ImpliedVolatilityBlack76(CP, valFrwd, K, T, frwdStock)
                plt.plot(TMat, IV * 100.0)
                legend.append('eta={0}'.format(etaTemp))
            plt.legend(legend)
            plot_placeholder.pyplot(fig2)

        # Effect of sigma
        if paremetereffect == "sigma":
            plt.title("Effect of sigma on Implied Volatility")
            plot_placeholder = st.empty()
            fig3=plt.figure(3)
            plt.grid()
            plt.xlabel('maturity, T')
            plt.ylabel('implied volatility')
            legend = []
            for sigmaTemp in sigmaV:
                IV = np.zeros([len(TMat), 1])
                for idx in range(0, len(TMat)):
                    T = TMat[idx]
                    val = BSHWOptionPrice(CP, S0, K, P0T(T), T, eta, sigmaTemp, rho, lambd)
                    frwdStock = S0 / P0T(T)
                    valFrwd = val / P0T(T)
                    IV[idx] = ImpliedVolatilityBlack76(CP, valFrwd, K, T, frwdStock)
                plt.plot(TMat, IV * 100.0)
                legend.append('sigma={0}'.format(sigmaTemp))
            plt.legend(legend)
            plot_placeholder.pyplot(fig3)
        # Effect of rho
        if paremetereffect == "rho":
            plt.title("Effect of rho on Implied Volatility")
            plot_placeholder = st.empty()
            fig4=plt.figure(4)
            plt.grid()
            plt.xlabel('maturity, T')
            plt.ylabel('implied volatility')
            legend = []
            for rhoTemp in rhoV:
                IV = np.zeros([len(TMat), 1])
                for idx in range(0, len(TMat)):
                    T = TMat[idx]
                    val = BSHWOptionPrice(CP, S0, K, P0T(T), T, eta, sigma, rhoTemp, lambd)
                    frwdStock = S0 / P0T(T)
                    valFrwd = val / P0T(T)
                    IV[idx] = ImpliedVolatilityBlack76(CP, valFrwd, K, T, frwdStock)
                plt.plot(TMat, IV * 100.0)
                legend.append('rho={0}'.format(rhoTemp))
            plt.legend(legend)
            plot_placeholder.pyplot(fig4)

#------------------------------------------------------------------------------------------------------------------------------------
if st.session_state["current_page"] == "hestonhw":
    st.title("Heston Hull White Model")
    st.write("You selected Heston Hull White model.")
    st.write("This model is used for option pricing in a stochastic volatility and interest rate environment.")
    st.write("You can use this model to analyze the impact of stochastic volatility and interest rates on option prices.")


    option_type = {"Call": OptionType.CALL, "Put": OptionType.PUT}
    slected_option = st.sidebar.selectbox("Select the option type", option_type.keys())
    CP = option_type[slected_option]

    NoOfPaths = st.sidebar.slider("Number of paths", min_value=100, max_value=100000, value=10000, step=100)
    NoOfSteps = st.sidebar.slider("Number of steps", min_value=1, max_value=1000, value=500, step=1)

    # HW model settings
    lambd = st.sidebar.number_input("Enter the mean reversion speed (lambda)", value=1.12, min_value=0.01, max_value=5.0, step=0.01)
    eta = st.sidebar.number_input("Enter the volatility of the short rate (eta)", value=0.01, min_value=0.01, max_value=1.0, step=0.01)
    S0 = st.sidebar.number_input("Enter the initial stock price (S0)", value=100.0, min_value=1.0, max_value=1000.0, step=1.0)
    T = st.sidebar.number_input("Enter the time to maturity (T)", value=15.0, min_value=0.01, max_value=100.0, step=0.01)
    r = st.sidebar.number_input("Enter the risk-free interest rate (r)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)

    # Strike range
    K = np.linspace(.01, 1.8 * S0 * np.exp(r * T), 20)
    K = np.array(K).reshape([len(K), 1])

    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-r * T)

    # Settings for the COS method
    N = 2000
    L = 15
    # Characteristic function of the Heston model + the COS method
    st.write("perameter settings of Heston Hull White model")
    gamma = st.sidebar.number_input("Enter the volatility of the stock (gamma)", value=0.3, min_value=0.01, max_value=1.0, step=0.01)
    vbar =st.sidebar.number_input("Enter the long-term volatility (vbar)", value=0.05, min_value=0.01, max_value=1.0, step=0.01)
    v0 = st.sidebar.number_input("Enter the initial volatility (v0)", value=0.02, min_value=0.01, max_value=1.0, step=0.01)
    rhoxr = st.sidebar.number_input("Enter the correlation between stock and interest rate (rhoxr)", value=0.5, min_value=-1.0, max_value=1.0, step=0.01)

    rhoxv = st.sidebar.number_input("Enter the correlation between stock and volatility (rhoxv)", value=-0.8, min_value=-1.0, max_value=1.0, step=0.01)
    kappa = st.sidebar.number_input("Enter the mean reversion speed of volatility (kappa)", value=0.5, min_value=0.01, max_value=5.0, step=0.01)

    np.random.seed(1)
    paths = H1_HW_cos_mc.GeneratePathsHestonHWEuler(NoOfPaths, NoOfSteps, P0T, T, S0, kappa, gamma, rhoxr, rhoxv, vbar, v0, lambd,
                                       eta)
    S = paths["S"]
    M_t = paths["M_t"]
    st.write("martingle check of the stock price with Eluer method")

    st.success(np.mean(S[:, -1] / M_t[:, -1]))
    valueOptMC = H1_HW_cos_mc.EUOptionPriceFromMCPathsGeneralizedStochIR(CP, S[:, -1], K, T, M_t[:, -1])

    np.random.seed(1)
    pathsExact = H1_HW_cos_mc.GeneratePathsHestonHW_AES(NoOfPaths, NoOfSteps, P0T, T, S0, kappa, gamma, rhoxr, rhoxv, vbar, v0,
                                           lambd, eta)
    S_ex = pathsExact["S"]
    M_t_ex = pathsExact["M_t"]
    valueOptMC_ex = H1_HW_cos_mc.EUOptionPriceFromMCPathsGeneralizedStochIR(CP, S_ex[:, -1], K, T, M_t_ex[:, -1])
    st.write("martingle check of the stock price with AES method")
    st.success(np.mean(S_ex[:, -1] / M_t_ex[:, -1]))
    plot_placeholder= st.empty()
    fig1=plt.figure(1)
    plt.title("Heston Hull White Model")
    plt.plot(K, valueOptMC)
    plt.plot(K, valueOptMC_ex, '.k')
    plt.ylim([0.0, 110.0])

    # The COS method
    cf2 = H1_HW_cos_mc.ChFH1HWModel(P0T, lambd, eta, T, kappa, gamma, vbar, v0, rhoxv, rhoxr)
    u = np.array([1.0, 2.0, 3.0,4.0])
    st.write(cf2(u))
    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf2, CP, S0, T, K, N, L, P0T(T))
    plt.plot(K, valCOS, '--r')
    plt.legend(['Euler', 'AES', 'COS'])
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('EU Option Value, K')
    plot_placeholder.pyplot(fig1)
    st.write("Value from the COS method:")
    st.success(valCOS)

#--------------------------------------------------------------------------------------------------------------------
if st.session_state["current_page"] == "szhw":
    st.title("Schobel-Zhu Hull White Model")
    st.write("You selected Schobel-Zhu Hull White model.")
    st.write("This model is used for option pricing in a stochastic volatility and interest rate environment.")
    st.write("You can use this model to analyze the impact of stochastic volatility and interest rates on option prices.")

    option_type = {"Call": OptionType.CALL, "Put": OptionType.PUT}
    slected_option = st.sidebar.selectbox("Select the option type", option_type.keys())
    CP = option_type[slected_option]

    # HW model settings
    st.sidebar.write("HW model settings")
    lambd = st.sidebar.number_input("Enter the mean reversion speed (lambda)", value= 0.425, min_value=0.01, max_value=1.0, step=0.01)
    eta = st.sidebar.number_input("Enter the volatility of the short rate (eta)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
    S0 = st.sidebar.number_input("Enter the initial stock price (S0)", value=100.0, min_value=1.0, max_value=1000.0, step=1.0)
    T = st.sidebar.number_input("Enter the time to maturity (T)", value=5.0, min_value=0.01, max_value=10.0, step=0.01)

    # The SZHW model
    st.sidebar.write("SZHW model settings")
    sigma0 = st.sidebar.number_input("Enter the initial volatility (sigma0)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
    gamma = st.sidebar.number_input("Enter the volatility of the stock (gamma)", value=0.11, min_value=0.01, max_value=1.0, step=0.01)
    Rrsigma = st.sidebar.number_input("Enter the correlation between stock and interest rate (Rrsigma)", value=0.32, min_value=-1.0, max_value=1.0, step=0.01)
    Rxsigma = st.sidebar.number_input("Enter the correlation between stock and volatility (Rxsigma)", value=-0.42, min_value=-1.0, max_value=1.0, step=0.01)
    Rxr = st.sidebar.number_input("Enter the correlation between stock and interest rate (Rxr)", value=0.3, min_value=-1.0, max_value=1.0, step=0.01)
    kappa = st.sidebar.number_input("Enter the mean reversion speed of volatility (kappa)", value= 0.4, min_value=0.01, max_value=1.0, step=0.01)
    sigmabar = st.sidebar.number_input("Enter the long-term volatility (sigmabar)", value=0.05, min_value=0.01, max_value=1.0, step=0.01)

    # Strike range
    K = np.linspace(40, 200.0, 20)
    K = np.array(K).reshape([len(K), 1])

    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.025 * T)

    # Forward stock
    frwdStock = S0 / P0T(T)

    # Settings for the COS method
    N = 2000
    L = 10
    plot_placeholder = st.empty()

    co1, col2 = st.columns(2)
    with co1:
        paremetereffect = co1.selectbox("Select the parameter to vary", ["gamma", "kappa", "rhoxsigma", "sigmabar"])
    if paremetereffect == "gamma":
        with col2:
            gamma_add = col2.number_input("Enter the value of gamma", min_value=0.0, max_value=1.0, value=0.05,

                                          step=0.01)
            gammaV = [0.1, 0.2, 0.3, 0.4]
            gammaV.append(gamma_add)
    if paremetereffect == "kappa":
        with col2:
            kappa_add = st.number_input("Enter the value of kappa", min_value=0.0, max_value=1.0, value=0.05,
                                        step=0.01)
            kappaV = [0.05, 0.2, 0.3, 0.4]
            kappaV.append(kappa_add)
    if paremetereffect == "rhoxsigma":
        with col2:
            Rxsigma_add = st.number_input("Enter the value of Rxsigma", min_value=-1.0, max_value=1.0, value=0.05,
                                          step=0.01)
            RxsigmaV = [-0.75, -0.25, 0.25, 0.75]
            RxsigmaV.append(Rxsigma_add)

    if paremetereffect == "sigmabar":
        with col2:
            sigmabar_add = st.number_input("Enter the value of sigmabar", min_value=0.0, max_value=1.0, value=0.05,
                                           step=0.01)
            sigmabarV = [0.1, 0.2, 0.3, 0.4]
            sigmabarV.append(sigmabar_add)

    # Effect of gamma
    if paremetereffect == "gamma":
        plt.title("Effect of gamma on Implied Volatility")
        fig1=plt.figure(1)
        plt.grid()
        plt.xlabel('strike, K')
        plt.ylabel('implied volatility')
        legend = []
        for gammaTemp in gammaV:
            # Evaluate the SZHW model
            cf = lambda u: SZHW_iV.ChFSZHW(u, P0T, sigma0, T, lambd, gammaTemp, Rxsigma, Rrsigma, Rxr, eta, kappa,
                                           sigmabar)

            # The COS method
            valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L, P0T(T))
            valCOSFrwd = valCOS / P0T(T)
            # Implied volatilities
            IV = np.zeros([len(K), 1])
            for idx in range(0, len(K)):
                IV[idx] = ImpliedVolatilityBlack76(CP, valCOSFrwd[idx], K[idx], T, frwdStock)
            plt.plot(K, IV * 100.0)
            legend.append('gamma={0}'.format(gammaTemp))
        plt.legend(legend)
        plot_placeholder.pyplot(fig1)
    # Effect of kappa
    if paremetereffect == "kappa":
        plt.title("Effect of kappa on Implied Volatility")
        fig2=plt.figure(2)
        plt.grid()
        plt.xlabel('strike, K')
        plt.ylabel('implied volatility')
        legend = []
        for kappaTemp in kappaV:
            # Evaluate the SZHW model
            cf = lambda u: SZHW_iV.ChFSZHW(u, P0T, sigma0, T, lambd, gamma, Rxsigma, Rrsigma, Rxr, eta, kappaTemp,
                                           sigmabar)

            # The COS method
            valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L, P0T(T))
            valCOSFrwd = valCOS / P0T(T)
            # Implied volatilities
            IV = np.zeros([len(K), 1])
            for idx in range(0, len(K)):
                IV[idx] = ImpliedVolatilityBlack76(CP, valCOSFrwd[idx], K[idx], T, frwdStock)
            plt.plot(K, IV * 100.0)
            legend.append('kappa={0}'.format(kappaTemp))
        plt.legend(legend)
        plot_placeholder.pyplot(fig2)

    # Effect of Rhoxsigma
    if paremetereffect == "rhoxsigma":
        plt.title("Effect of Rxsigma on Implied Volatility")
        fig3=plt.figure(3)
        plt.grid()
        plt.xlabel('strike, K')
        plt.ylabel('implied volatility')
        legend = []
        for RxsigmaTemp in RxsigmaV:
            # Evaluate the SZHW model
            cf = lambda u: SZHW_iV.ChFSZHW(u, P0T, sigma0, T, lambd, gamma, RxsigmaTemp, Rrsigma, Rxr, eta, kappa,
                                           sigmabar)

            # The COS method
            valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L, P0T(T))
            valCOSFrwd = valCOS / P0T(T)
            # Implied volatilities
            IV = np.zeros([len(K), 1])
            for idx in range(0, len(K)):
                IV[idx] = ImpliedVolatilityBlack76(CP, valCOSFrwd[idx], K[idx], T, frwdStock)
            plt.plot(K, IV * 100.0)
            legend.append('Rxsigma={0}'.format(RxsigmaTemp))
        plt.legend(legend)
        plot_placeholder.pyplot(fig3)



    # Effect of sigmabar
    if paremetereffect == "sigmabar":
        plt.title("Effect of sigmabar on Implied Volatility")
        fig4=plt.figure(4)
        plt.grid()
        plt.xlabel('strike, K')
        plt.ylabel('implied volatility')
        legend = []
        for sigmabarTemp in sigmabarV:
            # Evaluate the SZHW model
            cf = lambda u: SZHW_iV.ChFSZHW(u, P0T, sigma0, T, lambd, gamma, Rxsigma, Rrsigma, Rxr, eta, kappa,
                                           sigmabarTemp)

            # The COS method
            valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L, P0T(T))
            valCOSFrwd = valCOS / P0T(T)
            # Implied volatilities
            IV = np.zeros([len(K), 1])
            for idx in range(0, len(K)):
                IV[idx] = ImpliedVolatilityBlack76(CP, valCOSFrwd[idx], K[idx], T, frwdStock)
            plt.plot(K, IV * 100.0)
            legend.append('sigmabar={0}'.format(sigmabarTemp))
        plt.legend(legend)
        plot_placeholder.pyplot(fig4)

#---------------------------------------------------------------------------------------------------------------------------
if st.session_state["current_page"]=="diverficationhw":
    st.title("Diverfication of Hull White Model")
    st.write("You selected Diverfication of Hull White model.")
    st.subheader("Portfolio consiste of one stock and one ZCB bond")
    # HW model settings
    st.sidebar.write("HW model settings")
    lambd = st.sidebar.number_input("Enter the mean reversion speed (lambda)", value=1.12, min_value=0.01, max_value=5.0, step=0.01)
    eta = st.sidebar.number_input("Enter the volatility of the short rate (eta)", value=0.02, min_value=0.01, max_value=1.0, step=0.01)
    S0 = st.sidebar.number_input("Enter the initial stock price (S0)", value=100.0, min_value=1.0, max_value=1000.0, step=1.0)

    # Fixed mean reversion parameter
    kappa = 0.5
    # Diversification product
    st.sidebar.write("Diversification product settings")
    T = st.sidebar.number_input("Enter the time to maturity (T)", value=9.0, min_value=0.01, max_value=10.0, step=0.01)
    T1 = st.sidebar.number_input("Enter the time to maturity of the ZCB (T1)", value=10.0, min_value=0.01, max_value=10.0, step=0.01)

    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.033 * T)

    # Range of the waiting factor
    omegaV = np.linspace(-3.0, 3.0, 50)
    # Monte Carlo setting
    NoOfPaths = 5000
    NoOfSteps = int(100 * T)

    st.write("""The SZHW model parameters
     The parameters can be obtained by running the calibration of the SZHW model and with
    varying the correlation rhoxr.

    parameters = [{"Rxr": 0.0, "sigmabar": 0.167, "gamma": 0.2, "Rxsigma": -0.850, "Rrsigma": -0.008, "kappa": 0.5,
                   "sigma0": 0.035},
                  {"Rxr": -0.7, "sigmabar": 0.137, "gamma": 0.236, "Rxsigma": -0.381, "Rrsigma": -0.339, "kappa": 0.5,
                   "sigma0": 0.084},
                  {"Rxr": 0.7, "sigmabar": 0.102, "gamma": 0.211, "Rxsigma": -0.850, "Rrsigma": -0.340, "kappa": 0.5,
                   "sigma0": 0.01}]
    """)
    parameters = [{"Rxr": 0.0, "sigmabar": 0.167, "gamma": 0.2, "Rxsigma": -0.850, "Rrsigma": -0.008, "kappa": 0.5,
                   "sigma0": 0.035},
                  {"Rxr": -0.7, "sigmabar": 0.137, "gamma": 0.236, "Rxsigma": -0.381, "Rrsigma": -0.339, "kappa": 0.5,
                   "sigma0": 0.084},
                  {"Rxr": 0.7, "sigmabar": 0.102, "gamma": 0.211, "Rxsigma": -0.850, "Rrsigma": -0.340, "kappa": 0.5,
                   "sigma0": 0.01}]
    legend = []
    for (idx, par) in enumerate(parameters):
        sigma0 = par["sigma0"]
        gamma = par["gamma"]
        Rrsigma = par["Rrsigma"]
        Rxsigma = par["Rxsigma"]
        Rxr = par["Rxr"]
        sigmabar = par["sigmabar"]

        # Generate MC paths
        np.random.seed(1)
        paths = Diverfication_product.GeneratePathsSZHWEuler(NoOfPaths, NoOfSteps, P0T, T, S0, sigma0, sigmabar, kappa, gamma, lambd, eta,
                                       Rxsigma, Rxr, Rrsigma)
        S = paths["S"]
        M = paths["M_t"]
        R = paths["R"]

        S_T = S[:, -1]
        R_T = R[:, -1]
        M_T = M[:, -1]
        value_0 = Diverfication_product.DiversifcationPayoff(P0T, S_T, S0, R_T, M_T, T, T1, lambd, eta, omegaV)
        st.write("reference with rho=0.0")
        # reference with rho=0.0
        if Rxr == 0.0:
            refR0 = value_0

        fig1=plt.figure(1)
        plt.title("diversification product ptice")
        plt.plot(omegaV, value_0)
        plt.xlabel('omega')
        plt.ylabel('diversification product price')


        legend.append('par={0}'.format(idx))


        fig2=plt.figure(2)
        plt.title("relative correlation effect ")
        plt.xlabel('omega')
        plt.ylabel('relative correlation effect')
        plt.plot(omegaV, value_0 / refR0)

    fig1=plt.figure(1)
    plt.grid()
    plt.legend(legend)
    plt.tight_layout()
    st.pyplot(fig1)

    fig2= plt.figure(2)
    plt.grid()
    plt.legend(legend)
    plt.tight_layout()
    st.pyplot(fig2)
