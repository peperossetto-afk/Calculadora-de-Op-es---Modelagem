# =============================================================================
# CALCULADORA DE OPÇÕES FINANCEIRAS — DESIGN PROFISSIONAL
# Curso de Gestão de Riscos e Derivativos
#
# Métodos implementados:
#   - Black-Scholes (opções europeias)
#   - Simulação de Monte Carlo (europeias e asiáticas)
#   - Árvore Binomial CRR (europeias e americanas)
#   - Volatilidade Implícita (Newton-Raphson e Bisseção)
#
# Instalação:
#   pip install numpy pandas matplotlib scipy yfinance streamlit
#
# Execução:
#   streamlit run calculadora_opcoes.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# ─── TEMA VISUAL ─────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset geral ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: #080a0e !important;
    color: #c8d4e8 !important;
}

/* ── Fundo principal ── */
.stApp { background: #080a0e; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1017 !important;
    border-right: 1px solid #1c2333 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem; }

/* ── Títulos ── */
h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; font-size: 1.7rem !important; color: #e8eef8 !important; letter-spacing: -0.02em !important; }
h2 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; font-size: 1.1rem !important; color: #a0b0cc !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
h3 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; font-size: 0.9rem !important; color: #7890b0 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; }

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: #0f1520 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; letter-spacing: 0.12em !important; color: #506080 !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.5rem !important; font-weight: 700 !important; color: #e0e8f8 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background: #0f1520 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 4px !important;
    color: #c8d4e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #00c97a !important;
    box-shadow: 0 0 0 2px #00c97a22 !important;
}

/* ── Botão principal ── */
.stButton > button {
    background: linear-gradient(135deg, #00c97a 0%, #00a862 100%) !important;
    color: #060c10 !important;
    border: none !important;
    border-radius: 5px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #00c97a !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] span { font-size: 0.8rem !important; color: #8099bb !important; }

/* ── Tabelas ── */
[data-testid="stTable"] table {
    background: #0f1520 !important;
    border-collapse: collapse !important;
    font-size: 0.78rem !important;
    width: 100% !important;
}
[data-testid="stTable"] th {
    background: #141c28 !important;
    color: #506080 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid #1c2333 !important;
}
[data-testid="stTable"] td {
    padding: 7px 14px !important;
    border-bottom: 1px solid #141c28 !important;
    color: #c8d4e8 !important;
}

/* ── Alertas / Info ── */
[data-testid="stAlert"] {
    background: #0f1520 !important;
    border-radius: 5px !important;
    border-left: 3px solid #00c97a !important;
    font-size: 0.8rem !important;
}
.stWarning { border-left-color: #f5a623 !important; }
.stError   { border-left-color: #ff4d6a !important; }

/* ── Divisores ── */
hr { border-color: #1c2333 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1017 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 5px !important;
}
[data-testid="stExpander"] summary { font-size: 0.78rem !important; color: #7890b0 !important; letter-spacing: 0.06em !important; }

/* ── Success / spinner ── */
[data-testid="stSpinner"] { color: #00c97a !important; }
[data-testid="stSuccess"] { background: #001f10 !important; border-left: 3px solid #00c97a !important; color: #00c97a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080a0e; }
::-webkit-scrollbar-thumb { background: #1c2333; border-radius: 2px; }
</style>
"""

# Paleta para matplotlib
MPLOT = {
    "bg":      "#080a0e",
    "bg2":     "#0f1520",
    "border":  "#1c2333",
    "text":    "#c8d4e8",
    "muted":   "#506080",
    "green":   "#00c97a",
    "red":     "#ff4d6a",
    "amber":   "#f5a623",
    "blue":    "#3b82f6",
    "cyan":    "#22d3ee",
    "purple":  "#a855f7",
}

def apply_dark_theme(fig, axes=None):
    """Aplica o tema dark profissional em figuras matplotlib."""
    fig.patch.set_facecolor(MPLOT["bg2"])
    if axes is None:
        axes = fig.get_axes()
    elif not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(MPLOT["bg2"])
        ax.tick_params(colors=MPLOT["muted"], labelsize=8)
        ax.xaxis.label.set_color(MPLOT["muted"])
        ax.yaxis.label.set_color(MPLOT["muted"])
        ax.title.set_color(MPLOT["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(MPLOT["border"])
        ax.grid(color=MPLOT["border"], linewidth=0.5, alpha=0.6)


# =============================================================================
# 1. FUNÇÕES DE DADOS — Yahoo Finance
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def baixar_dados_completos(ticker: str):
    """
    Baixa preços históricos e informações do ativo via yfinance.
    Cache de 5 minutos para evitar requisições repetidas.

    Retorna:
        precos   : Series com fechamentos ajustados (2 anos)
        info     : dict com metadados do ativo (nome, setor, etc.)
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="2y", auto_adjust=True)
    if hist.empty:
        return None, None
    precos = hist["Close"].dropna()

    try:
        info = t.info
    except Exception:
        info = {}

    return precos, info


def calcular_parametros(precos):
    """
    Calcula preço atual e volatilidade histórica anualizada.

    Retornos logarítmicos: r_t = ln(S_t / S_{t-1})
    Vol anualizada:        σ = std(r_t) × √252
    """
    retornos = np.log(precos / precos.shift(1)).dropna()
    S0    = float(precos.iloc[-1])
    sigma = float(retornos.std() * np.sqrt(252))
    return S0, sigma, retornos


def formatar_numero(v, prefixo="R$ ", decimais=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1e12:
        return f"{prefixo}{v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{prefixo}{v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{prefixo}{v/1e6:.2f}M"
    return f"{prefixo}{v:,.{decimais}f}"


# =============================================================================
# 2. BLACK-SCHOLES
# =============================================================================

def black_scholes(S0, K, r, T, sigma, tipo="call"):
    """
    Preço teórico Black-Scholes e gregas básicas.

    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 − σ·√T
    Call: C = S·N(d1) − K·e^{−rT}·N(d2)
    Put : P = K·e^{−rT}·N(−d2) − S·N(−d1)
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo.lower() == "call":
        preco = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega_ = S0 * np.sqrt(T) * norm.pdf(d1) / 100        # por 1% de vol
    if tipo.lower() == "call":
        theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {
        "preco": preco, "d1": d1, "d2": d2,
        "delta": delta, "gamma": gamma, "vega": vega_, "theta": theta
    }


def vega_bs(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * norm.pdf(d1)


# =============================================================================
# 3. MONTE CARLO
# =============================================================================

def simular_caminhos(S0, r, sigma, T, passos, n_sim, seed=42):
    np.random.seed(seed)
    dt = T / passos
    caminhos = np.zeros((passos + 1, n_sim))
    caminhos[0] = S0
    for t in range(1, passos + 1):
        Z = np.random.standard_normal(n_sim)
        caminhos[t] = caminhos[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    return caminhos


def monte_carlo_europeia(S0, K, r, T, sigma, tipo="call", n_sim=10000, passos=252):
    caminhos = simular_caminhos(S0, r, sigma, T, passos, n_sim)
    ST = caminhos[-1]
    payoffs = np.maximum(ST - K, 0) if tipo.lower() == "call" else np.maximum(K - ST, 0)
    preco = np.exp(-r * T) * np.mean(payoffs)
    return preco, caminhos, payoffs


def monte_carlo_asiatica(S0, K, r, T, sigma, tipo="call", n_sim=10000, passos=252):
    caminhos = simular_caminhos(S0, r, sigma, T, passos, n_sim)
    medias = np.mean(caminhos[1:], axis=0)
    payoffs = np.maximum(medias - K, 0) if tipo.lower() == "call" else np.maximum(K - medias, 0)
    preco = np.exp(-r * T) * np.mean(payoffs)
    return preco, caminhos, medias


# =============================================================================
# 4. ÁRVORE BINOMIAL (CRR)
# =============================================================================

def arvore_binomial(S0, K, r, T, sigma, tipo="call", estilo="europeia", n_passos=100):
    dt = T / n_passos
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)

    j = np.arange(n_passos + 1)
    precos_finais = S0 * (u ** j) * (d ** (n_passos - j))
    valores = (np.maximum(precos_finais - K, 0)
               if tipo.lower() == "call"
               else np.maximum(K - precos_finais, 0))

    for i in range(n_passos - 1, -1, -1):
        valores = np.exp(-r * dt) * (p * valores[1:i+2] + (1-p) * valores[0:i+1])
        if estilo.lower() == "americana":
            precos_nivel = S0 * (u ** np.arange(i+1)) * (d ** (i - np.arange(i+1)))
            exercicio = (np.maximum(precos_nivel - K, 0)
                         if tipo.lower() == "call"
                         else np.maximum(K - precos_nivel, 0))
            valores = np.maximum(valores, exercicio)

    return float(valores[0])


# =============================================================================
# 5. VOLATILIDADE IMPLÍCITA
# =============================================================================

def newton_raphson(preco_mercado, S0, K, r, T, tipo="call",
                   sigma_inicial=0.3, tol=1e-8, max_iter=100):
    sigma = sigma_inicial
    historico = [sigma]
    for _ in range(max_iter):
        bs  = black_scholes(S0, K, r, T, sigma, tipo)
        v   = vega_bs(S0, K, r, T, sigma)
        if abs(v) < 1e-10:
            return None, historico, False
        sigma_novo = sigma - (bs["preco"] - preco_mercado) / v
        sigma_novo = max(sigma_novo, 1e-6)
        historico.append(sigma_novo)
        if abs(sigma_novo - sigma) < tol:
            return sigma_novo, historico, True
        sigma = sigma_novo
    return sigma, historico, False


def bissecao(preco_mercado, S0, K, r, T, tipo="call",
             sigma_min=0.001, sigma_max=5.0, tol=1e-8, max_iter=200):
    f = lambda s: black_scholes(S0, K, r, T, s, tipo)["preco"] - preco_mercado
    f_min = f(sigma_min)
    if f_min * f(sigma_max) > 0:
        return None, [], False
    historico = []
    for _ in range(max_iter):
        sigma_mid = (sigma_min + sigma_max) / 2
        f_mid = f(sigma_mid)
        historico.append(sigma_mid)
        if abs(f_mid) < tol or (sigma_max - sigma_min) / 2 < tol:
            return sigma_mid, historico, True
        if f_min * f_mid < 0:
            sigma_max = sigma_mid
        else:
            sigma_min = sigma_mid
            f_min = f_mid
    return sigma_mid, historico, False


# =============================================================================
# 6. GRÁFICOS
# =============================================================================

def grafico_ativo(precos, ticker, S0, sigma_hist):
    """
    Painel de 2 subplots: preço histórico + histograma de retornos.
    """
    retornos = np.log(precos / precos.shift(1)).dropna()

    fig = plt.figure(figsize=(11, 4.5), facecolor=MPLOT["bg2"])
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.04)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    apply_dark_theme(fig, [ax1, ax2])

    # ── Preço histórico ──
    datas  = precos.index
    precos_arr = precos.values
    ax1.fill_between(datas, precos_arr, alpha=0.12, color=MPLOT["green"])
    ax1.plot(datas, precos_arr, color=MPLOT["green"], lw=1.5)
    ax1.axhline(S0, color=MPLOT["amber"], lw=1, linestyle="--", alpha=0.7,
                label=f"Último: {S0:.2f}")
    ax1.set_title(f"{ticker.upper()}  —  Preço Histórico (2 anos)", fontsize=9,
                  color=MPLOT["text"], pad=8)
    ax1.set_ylabel("Preço", fontsize=8)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=8, framealpha=0, labelcolor=MPLOT["amber"])
    # Sombra de MA50
    ma50 = precos.rolling(50).mean()
    ax1.plot(datas, ma50.values, color=MPLOT["purple"], lw=1, alpha=0.5,
             linestyle="--", label="MA50")

    # ── Histograma de retornos ──
    mu, std = retornos.mean(), retornos.std()
    ax2.hist(retornos, bins=55, density=True, color=MPLOT["blue"],
             edgecolor=MPLOT["bg2"], alpha=0.75)
    x = np.linspace(retornos.min(), retornos.max(), 300)
    ax2.plot(x, norm.pdf(x, mu, std), color=MPLOT["green"], lw=1.5,
             label="Normal ajust.")
    ax2.set_title(f"Retornos Log.  σ={sigma_hist*100:.1f}%/ano", fontsize=9,
                  color=MPLOT["text"], pad=8)
    ax2.set_ylabel("Densidade", fontsize=8)
    ax2.legend(fontsize=8, framealpha=0, labelcolor=MPLOT["green"])

    fig.tight_layout()
    return fig


def grafico_payoff(S0, K, bs_result, tipo, preco_opcao):
    """
    Gráfico de payoff + diagrama de lucro/perda ao vencimento.
    """
    ST     = np.linspace(S0 * 0.5, S0 * 1.6, 400)
    payoff = (np.maximum(ST - K, 0) if tipo.lower() == "call"
              else np.maximum(K - ST, 0))
    pl     = payoff - preco_opcao

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=MPLOT["bg2"])
    apply_dark_theme(fig, ax)

    ax.fill_between(ST, payoff, alpha=0.10, color=MPLOT["green"])
    ax.plot(ST, payoff, color=MPLOT["green"], lw=2, label="Payoff bruto")
    ax.plot(ST, pl,     color=MPLOT["cyan"],  lw=1.5, linestyle="--",
            label=f"P&L (prêmio={preco_opcao:.4f})")
    ax.axhline(0, color=MPLOT["border"], lw=1)
    ax.axvline(K,  color=MPLOT["red"],   lw=1, linestyle="--",
               label=f"Strike K={K:.2f}")
    ax.axvline(S0, color=MPLOT["amber"], lw=1, linestyle="--",
               label=f"S₀={S0:.2f}")

    ax.set_xlabel("Preço no Vencimento  S_T", fontsize=8)
    ax.set_ylabel("Valor", fontsize=8)
    ax.set_title(f"Payoff no Vencimento — {tipo.upper()}", fontsize=9,
                 color=MPLOT["text"], pad=8)
    ax.legend(fontsize=8, framealpha=0, ncol=2)
    fig.tight_layout()
    return fig


def grafico_monte_carlo(caminhos, K, tipo, n_mostrar=60):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=MPLOT["bg2"])
    apply_dark_theme(fig, ax)

    n = min(n_mostrar, caminhos.shape[1])
    for i in range(n):
        ax.plot(caminhos[:, i], lw=0.6, alpha=0.35, color=MPLOT["blue"])
    ax.plot(caminhos.mean(axis=1), color=MPLOT["green"], lw=2,
            label="Média")
    ax.axhline(K, color=MPLOT["red"], lw=1.2, linestyle="--",
               label=f"K = {K:.2f}")

    ax.set_xlabel("Passos de Tempo", fontsize=8)
    ax.set_ylabel("Preço Simulado", fontsize=8)
    ax.set_title(f"Monte Carlo — {n_mostrar} trajetórias amostradas", fontsize=9,
                 color=MPLOT["text"], pad=8)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout()
    return fig


def grafico_convergencia_vi(hist_nr, hist_biss, vi_nr, vi_biss):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=MPLOT["bg2"])
    apply_dark_theme(fig, axes)

    for ax, hist, vi, cor, label in [
        (axes[0], hist_nr,   vi_nr,   MPLOT["cyan"],   "Newton-Raphson"),
        (axes[1], hist_biss, vi_biss, MPLOT["purple"], "Bisseção"),
    ]:
        if hist:
            ax.plot(range(len(hist)), hist, marker="o", ms=4,
                    color=cor, lw=1.5)
            if vi:
                ax.axhline(vi, linestyle="--", color=MPLOT["amber"], lw=1,
                           label=f"VI = {vi*100:.4f}%")
        ax.set_title(f"Convergência — {label}", fontsize=9,
                     color=MPLOT["text"], pad=8)
        ax.set_xlabel("Iteração", fontsize=8)
        ax.set_ylabel("σ estimado", fontsize=8)
        ax.legend(fontsize=8, framealpha=0)

    fig.tight_layout()
    return fig


def grafico_comparacao(bs_val, mc_val, bin_val):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=MPLOT["bg2"])
    apply_dark_theme(fig, ax)

    labels = ["Black-Scholes", "Monte Carlo", "Binomial"]
    values = [bs_val, mc_val, bin_val]
    cores  = [MPLOT["green"], MPLOT["cyan"], MPLOT["amber"]]

    bars = ax.barh(labels, values, color=cores, height=0.45, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(v + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", ha="left",
                color=MPLOT["text"], fontsize=9, fontweight="bold")

    ax.set_title("Comparação de Preços", fontsize=9, color=MPLOT["text"], pad=8)
    ax.set_xlabel("Preço da Opção", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# =============================================================================
# 7. INTERFACE STREAMLIT
# =============================================================================

def painel_ativo(ticker):
    """
    Renderiza o card de informações do ativo assim que o ticker é alterado.
    Usa cache para evitar múltiplas requisições ao Yahoo Finance.
    """
    with st.spinner(f"Carregando {ticker}..."):
        precos, info = baixar_dados_completos(ticker)

    if precos is None:
        st.error(f"Ticker **{ticker}** não encontrado. Verifique e tente novamente.")
        return None, None, None

    S0, sigma_hist, retornos = calcular_parametros(precos)

    # ── Linha superior: nome + preço ──
    nome = info.get("longName") or info.get("shortName") or ticker.upper()
    moeda = info.get("currency", "BRL")
    variacao_dia = info.get("regularMarketChangePercent", 0.0) or 0.0
    seta = "▲" if variacao_dia >= 0 else "▼"
    cor_var = "green" if variacao_dia >= 0 else "red"

    st.markdown(f"""
    <div style="background:#0d1017;border:1px solid #1c2333;border-radius:8px;padding:18px 22px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.5rem;color:#e8eef8;letter-spacing:-0.02em;">{ticker.upper()}</span>
                <span style="font-size:0.72rem;background:#1c2333;color:#8099bb;border-radius:3px;padding:2px 8px;margin-left:10px;letter-spacing:0.06em;">{moeda}</span>
                <div style="font-size:0.8rem;color:#506080;margin-top:4px;">{nome}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.8rem;color:#e8eef8;">{S0:.2f}</div>
                <div style="font-size:0.82rem;color:{'#00c97a' if variacao_dia >= 0 else '#ff4d6a'};font-weight:600;">{seta} {abs(variacao_dia):.2f}% hoje</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Grid de métricas ──
    c1, c2, c3, c4, c5 = st.columns(5)

    abertura  = info.get("regularMarketOpen")   or info.get("open")
    max_dia   = info.get("regularMarketDayHigh") or info.get("dayHigh")
    min_dia   = info.get("regularMarketDayLow")  or info.get("dayLow")
    volume    = info.get("regularMarketVolume")  or info.get("volume")
    mkt_cap   = info.get("marketCap")
    beta      = info.get("beta")
    pe        = info.get("trailingPE") or info.get("forwardPE")
    div_yield = info.get("dividendYield")
    high_52w  = info.get("fiftyTwoWeekHigh")
    low_52w   = info.get("fiftyTwoWeekLow")

    with c1:
        st.metric("Abertura",    f"{abertura:.2f}"  if abertura  else "—")
    with c2:
        st.metric("Máx. Dia",   f"{max_dia:.2f}"   if max_dia   else "—")
    with c3:
        st.metric("Mín. Dia",   f"{min_dia:.2f}"   if min_dia   else "—")
    with c4:
        vol_str = formatar_numero(volume, prefixo="") if volume else "—"
        st.metric("Volume",      vol_str)
    with c5:
        st.metric("Vol. Hist. σ", f"{sigma_hist*100:.1f}%  /ano")

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.metric("Market Cap",  formatar_numero(mkt_cap) if mkt_cap else "—")
    with c7:
        st.metric("Beta",        f"{beta:.2f}"   if beta      else "—")
    with c8:
        st.metric("P/L",         f"{pe:.2f}×"   if pe        else "—")
    with c9:
        dy = f"{div_yield*100:.2f}%" if div_yield else "—"
        st.metric("Div. Yield",  dy)
    with c10:
        rng = f"{low_52w:.2f} – {high_52w:.2f}" if high_52w and low_52w else "—"
        st.metric("Range 52 sem.", rng)

    # ── Gráfico histórico + retornos ──
    st.pyplot(grafico_ativo(precos, ticker, S0, sigma_hist))

    return S0, sigma_hist, precos


def main():
    st.set_page_config(
        page_title="Options Desk | Calculadora de Opções",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Injeta CSS
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
        <span style="font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:1.8rem;color:#e8eef8;letter-spacing:-0.03em;">
            <span style="color:#00c97a;">Δ</span> OPTIONS DESK
        </span>
        <span style="font-size:0.68rem;background:#001f10;color:#00c97a;border:1px solid #00c97a44;border-radius:3px;padding:2px 8px;letter-spacing:0.1em;">BLACK-SCHOLES</span>
        <span style="font-size:0.68rem;background:#0f1a2e;color:#3b82f6;border:1px solid #3b82f644;border-radius:3px;padding:2px 8px;letter-spacing:0.1em;">MONTE CARLO</span>
        <span style="font-size:0.68rem;background:#1c1205;color:#f5a623;border:1px solid #f5a62344;border-radius:3px;padding:2px 8px;letter-spacing:0.1em;">BINOMIAL CRR</span>
    </div>
    <div style="font-size:0.78rem;color:#506080;letter-spacing:0.06em;margin-bottom:20px;">
        GESTÃO DE RISCOS & DERIVATIVOS — PRECIFICAÇÃO PROFISSIONAL DE OPÇÕES
    </div>
    <hr style="border-color:#1c2333;margin-bottom:20px;">
    """, unsafe_allow_html=True)

    # =========================================================================
    # PAINEL LATERAL
    # =========================================================================
    with st.sidebar:
        st.markdown("### ◎ Ativo")
        ticker = st.text_input(
            "Ticker",
            value="PETR4.SA",
            placeholder="PETR4.SA  |  VALE3.SA  |  AAPL",
            help="Código do ativo no Yahoo Finance. Ações BR: adicione .SA",
        )
        usar_yahoo = st.checkbox("Buscar dados ao vivo (Yahoo Finance)", value=True)

        st.markdown("---")
        st.markdown("### ◎ Opção")

        tipo_payoff = st.selectbox("Payoff", ["Call", "Put"])
        tipo_opcao  = st.selectbox("Estilo", ["Europeia", "Americana", "Asiática"])
        metodo      = st.selectbox(
            "Método de Precificação",
            ["Black-Scholes", "Monte Carlo", "Árvore Binomial"],
        )

        st.markdown("---")
        st.markdown("### ◎ Parâmetros")

        S0_input    = st.number_input("S₀ — Preço Atual",              value=38.00, min_value=0.01, step=0.50, format="%.2f")
        K           = st.number_input("K  — Strike",                   value=40.00, min_value=0.01, step=0.50, format="%.2f")
        T           = st.number_input("T  — Prazo (anos)",             value=0.50,  min_value=0.01, step=0.05, format="%.4f")
        r_pct       = st.number_input("r  — Taxa livre de risco (% a.a.)", value=10.0, min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
        sigma_pct   = st.number_input("σ  — Volatilidade (% a.a.)",   value=30.0,  min_value=0.1, max_value=300.0, step=1.0, format="%.1f")

        r     = r_pct / 100.0
        sigma = sigma_pct / 100.0

        st.markdown("---")

        n_sim, n_passos = 10000, 100
        if metodo == "Monte Carlo":
            n_sim    = st.slider("Simulações", 1000, 50000, 10000, step=1000)
        if metodo == "Árvore Binomial":
            n_passos = st.slider("Passos (n)", 10, 500, 200, step=10)

        st.markdown("---")
        st.markdown("### ◎ Vol. Implícita")
        preco_mercado = st.number_input(
            "Preço de Mercado da Opção",
            value=0.00, min_value=0.00, step=0.01, format="%.4f",
            help="Informe o prêmio observado no mercado para calcular a VI. Deixe 0 para omitir.",
        )

        calcular = st.button("PRECIFICAR OPÇÃO  →")

    # =========================================================================
    # DADOS DO ATIVO — exibidos imediatamente ao digitar o ticker
    # =========================================================================
    S0     = S0_input          # padrão: valor manual
    sigma  = sigma_pct / 100.0

    if usar_yahoo and ticker.strip():
        resultado_yahoo = painel_ativo(ticker.strip().upper())
        if resultado_yahoo[0] is not None:
            S0_yahoo, sigma_yahoo, _precos = resultado_yahoo

            st.markdown("---")
            col_a, col_b = st.columns([1, 3])
            with col_a:
                usar_s0    = st.checkbox(f"Usar S₀ = {S0_yahoo:.2f} do Yahoo", value=True)
                usar_sigma = st.checkbox(f"Usar σ = {sigma_yahoo*100:.1f}%  do Yahoo", value=True)
            if usar_s0:
                S0 = S0_yahoo
            if usar_sigma:
                sigma = sigma_yahoo
            st.markdown("---")

    # =========================================================================
    # PRECIFICAÇÃO
    # =========================================================================
    if calcular:

        tipo_str = tipo_payoff.lower()

        # Verificação de compatibilidade
        avisos = []
        if metodo == "Black-Scholes" and tipo_opcao != "Europeia":
            avisos.append(f"⚠ Black-Scholes é exclusivo para opções **Europeias**. O resultado abaixo usa Monte Carlo.")
        if tipo_opcao == "Asiática" and metodo == "Árvore Binomial":
            avisos.append("⚠ Árvore Binomial não modela opções **Asiáticas**. Redirecionando para Monte Carlo.")
        for av in avisos:
            st.warning(av)

        # ── Cálculos ──
        bs_res = black_scholes(S0, K, r, T, sigma, tipo_str)

        mc_preco, mc_caminhos, mc_payoffs = monte_carlo_europeia(
            S0, K, r, T, sigma, tipo_str, n_sim=n_sim, passos=252
        )

        bin_preco = arvore_binomial(
            S0, K, r, T, sigma, tipo_str,
            "americana" if tipo_opcao == "Americana" else "europeia",
            n_passos
        )

        # ── Seção: Preços ──
        st.markdown("## 01 — Resultados da Precificação")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Black-Scholes",   f"R$ {bs_res['preco']:.4f}",
                      delta=f"d₁={bs_res['d1']:.4f}  d₂={bs_res['d2']:.4f}")
        with col2:
            delta_mc = f"{((mc_preco - bs_res['preco'])/bs_res['preco']*100):+.2f}% vs BS"
            st.metric("Monte Carlo",     f"R$ {mc_preco:.4f}", delta=delta_mc)
        with col3:
            delta_bin = f"{((bin_preco - bs_res['preco'])/bs_res['preco']*100):+.2f}% vs BS"
            st.metric(f"Árvore Binomial ({tipo_opcao})", f"R$ {bin_preco:.4f}", delta=delta_bin)

        # ── Gregas ──
        st.markdown("## 02 — Greeks (Black-Scholes)")

        cg1, cg2, cg3, cg4 = st.columns(4)
        with cg1: st.metric("Δ  Delta", f"{bs_res['delta']:+.4f}",
                             help="Sensibilidade do preço da opção à variação de S₀")
        with cg2: st.metric("Γ  Gamma", f"{bs_res['gamma']:.6f}",
                             help="Taxa de variação do Delta")
        with cg3: st.metric("ν  Vega",  f"{bs_res['vega']:.4f}  /1%σ",
                             help="Sensibilidade por 1% de variação na volatilidade")
        with cg4: st.metric("Θ  Theta", f"{bs_res['theta']:.4f}  /dia",
                             help="Decaimento temporal diário do prêmio")

        # ── Parâmetros utilizados ──
        with st.expander("▸ Parâmetros utilizados", expanded=False):
            df_params = pd.DataFrame({
                "Parâmetro": ["S₀", "K", "T (anos)", "r (% a.a.)", "σ (% a.a.)",
                              "Tipo Payoff", "Estilo", "Método"],
                "Valor": [f"{S0:.4f}", f"{K:.4f}", f"{T:.4f}",
                          f"{r*100:.2f}%", f"{sigma*100:.2f}%",
                          tipo_payoff, tipo_opcao, metodo],
            })
            st.table(df_params)

        # ── Gráficos ──
        st.markdown("## 03 — Análise Visual")

        gv1, gv2 = st.columns(2)
        with gv1:
            st.pyplot(grafico_payoff(S0, K, bs_res, tipo_str, bs_res["preco"]))
        with gv2:
            st.pyplot(grafico_comparacao(bs_res["preco"], mc_preco, bin_preco))

        st.pyplot(grafico_monte_carlo(mc_caminhos, K, tipo_opcao))

        # ── Volatilidade Implícita ──
        st.markdown("## 04 — Volatilidade Implícita")

        if preco_mercado > 0:
            vi_nr,   hist_nr,   conv_nr   = newton_raphson(preco_mercado, S0, K, r, T, tipo_str)
            vi_biss, hist_biss, conv_biss = bissecao(preco_mercado, S0, K, r, T, tipo_str)

            vi1, vi2 = st.columns(2)
            with vi1:
                if conv_nr:
                    st.metric("VI — Newton-Raphson",
                              f"{vi_nr*100:.4f}%",
                              delta=f"{len(hist_nr)} iterações")
                else:
                    st.warning("Newton-Raphson não convergiu.")
            with vi2:
                if conv_biss:
                    st.metric("VI — Bisseção",
                              f"{vi_biss*100:.4f}%",
                              delta=f"{len(hist_biss)} iterações")
                else:
                    st.warning("Bisseção não convergiu.")

            st.pyplot(grafico_convergencia_vi(hist_nr, hist_biss, vi_nr, vi_biss))

            # Tabela resumo
            vi_df = pd.DataFrame({
                "Método":                ["Newton-Raphson", "Bisseção"],
                "Volatilidade Implícita": [
                    f"{vi_nr*100:.4f}%"   if vi_nr   else "N/A",
                    f"{vi_biss*100:.4f}%" if vi_biss else "N/A",
                ],
                "Iterações": [
                    str(len(hist_nr))   if hist_nr   else "N/A",
                    str(len(hist_biss)) if hist_biss else "N/A",
                ],
                "Convergiu": ["Sim" if conv_nr else "Não",
                              "Sim" if conv_biss else "Não"],
            })
            st.table(vi_df)

        else:
            st.info("Informe o **Preço de Mercado da Opção** para calcular a Volatilidade Implícita.")

        # ── Exemplos do Case ──
        st.markdown("## 05 — Exemplos do Case")

        with st.expander("Exemplo 1 — Call Europeia PETR4 (BS / MC / Binomial)"):
            p1_bs  = black_scholes(38, 40, 0.10, 0.5, 0.30, "call")["preco"]
            p1_mc, _, _ = monte_carlo_europeia(38, 40, 0.10, 0.5, 0.30, "call", 10000)
            p1_bin = arvore_binomial(38, 40, 0.10, 0.5, 0.30, "call", "europeia", 200)
            st.table(pd.DataFrame({
                "Método": ["Black-Scholes", "Monte Carlo", "Binomial"],
                "Preço (R$)": [f"{p1_bs:.4f}", f"{p1_mc:.4f}", f"{p1_bin:.4f}"],
            }))

        with st.expander("Exemplo 2 — Put Americana vs Europeia (Árvore Binomial)"):
            p2_am = arvore_binomial(50, 55, 0.08, 1.0, 0.25, "put", "americana", 200)
            p2_eu, _, _ = (lambda r: (r["preco"], r["d1"], r["d2"]))(
                black_scholes(50, 55, 0.08, 1.0, 0.25, "put"))
            st.table(pd.DataFrame({
                "Tipo":   ["Put Americana (Binomial)", "Put Europeia (BS)", "Prêmio antecipação"],
                "Preço (R$)": [f"{p2_am:.4f}", f"{p2_eu:.4f}", f"{p2_am - p2_eu:.4f}"],
            }))

        with st.expander("Exemplo 3 — Call Asiática (Monte Carlo)"):
            p3_asia, _, _ = monte_carlo_asiatica(38, 40, 0.10, 0.5, 0.30, "call", 10000)
            p3_eu = black_scholes(38, 40, 0.10, 0.5, 0.30, "call")["preco"]
            st.table(pd.DataFrame({
                "Tipo":     ["Call Asiática (MC)", "Call Europeia (BS)", "Desconto asiático"],
                "Preço (R$)": [f"{p3_asia:.4f}", f"{p3_eu:.4f}", f"{p3_eu - p3_asia:.4f}"],
            }))

    else:
        # Estado inicial
        st.markdown("""
        <div style="background:#0d1017;border:1px dashed #1c2333;border-radius:8px;padding:40px;text-align:center;margin-top:16px;">
            <div style="font-size:2.5rem;opacity:0.12;margin-bottom:12px;">∂</div>
            <div style="font-size:0.85rem;color:#506080;line-height:1.9;">
                Configure os parâmetros no painel lateral<br>
                e clique em <strong style="color:#00c97a;">PRECIFICAR OPÇÃO →</strong> para calcular
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── PONTO DE ENTRADA ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()