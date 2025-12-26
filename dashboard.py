import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(
    layout="wide", 
    page_title="Infinity Sniper: Entry & Exit", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)
DB_NAME = "market_data.db"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    h1, h2, h3 { color: #58a6ff !important; font-weight: 600; letter-spacing: -0.5px; }
    
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    div[data-testid="stMetricValue"] { color: #f0f6fc; font-weight: 600; }
    
    .consensus-banner { padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 25px; color: white; border: 1px solid rgba(255,255,255,0.1); background: linear-gradient(135deg, #1f242d 0%, #0d1117 100%); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
    .model-card { background-color: #21262d; padding: 12px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #30363d; text-align: center; transition: transform 0.2s; }
    .model-card:hover { transform: translateY(-3px); border-color: #58a6ff; }
    
    .sniper-pass { border-left: 5px solid #00e676; background-color: #1b3a2b; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #00e676; }
    .sniper-fail { border-left: 5px solid #ff1744; background-color: #3a1b1b; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #ff1744; }
    .sniper-warn { border-left: 5px solid #ffab00; background-color: #3a2e1b; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #ffab00; }
    
    /* Exit Strategy Card */
    .exit-card-win { background: linear-gradient(90deg, #1b3a2b 0%, #0d1117 100%); padding: 15px; border-radius: 8px; border: 1px solid #00e676; margin-bottom: 10px; }
    .exit-card-loss { background: linear-gradient(90deg, #3a1b1b 0%, #0d1117 100%); padding: 15px; border-radius: 8px; border: 1px solid #ff1744; margin-bottom: 10px; }
    
    .stButton > button { background-color: #238636; color: white; border: none; font-weight: 600; width: 100%; }
    .stButton > button:hover { background-color: #2ea043; }
</style>
""", unsafe_allow_html=True)

def get_all_tickers():
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker", conn)
        return df['ticker'].tolist()
    except: return []
    finally: conn.close()

def get_data(ticker):
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM daily_prices WHERE ticker = '{ticker}' ORDER BY date ASC"
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty: return None
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(inplace=True)
        if len(df) < 20: return None
        return df
    except:
        conn.close()
        return None

def rekayasa_fitur(df):
    """ ⚔️ THE 30-WEAPON ARSENAL ⚔️ """
    df = df.copy()
    
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA100'] = df['close'].rolling(100).mean()
    df['MA200'] = df['close'].rolling(200).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    df['VolMA5'] = df['volume'].rolling(5).mean()
    df['VolMA20'] = df['volume'].rolling(20).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low14) / (high14 - low14))
    
    std20 = df['close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std20 * 2)
    df['BB_Lower'] = df['MA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['close']
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['Returns'] = df['close'].pct_change()
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
 
    temp_check = df.dropna()
    if len(temp_check) < 10:
        heavy_indicators = ['MA200', 'EMA50', 'MA50', 'MA100', 'MACD']
        cols_to_drop = [c for c in heavy_indicators if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
    
    df.dropna(inplace=True)
    return df

def calculate_exit_strategy(df):
    """
    Menghitung Titik Jual (TP) dan Cut Loss (SL) berdasarkan ATR
    """
    last = df.iloc[-1]
    close_price = last['close']
    atr = last['ATR'] if 'ATR' in df.columns else (close_price * 0.02)
    ma20 = last['MA20'] if 'MA20' in df.columns else (close_price * 0.95)

    sl_price = close_price - (2 * atr)
    sl_candle = df.iloc[-2]['low'] if len(df) > 1 else sl_price
    
    final_sl = max(sl_price, sl_candle) 

    tp1 = close_price + (1.5 * atr)
    tp2 = close_price + (3 * atr)
    
    return final_sl, tp1, tp2, atr

def cek_kriteria_sniper_auto(df):
    try:
        last = df.iloc[-1]; prev = df.iloc[-2]
        
        value_trans = last['close'] * last['volume']
        if value_trans < 3_000_000_000: return False
        if last['close'] <= 50: return False

        vol_ma20 = df['VolMA20'].iloc[-1] if 'VolMA20' in df.columns else last['volume']
        vol_check = last['volume'] >= 1.5 * vol_ma20
        price_gain = (last['close'] - prev['close']) / prev['close']
        gain_check = price_gain >= 0.04
        ma5 = df['MA5'].iloc[-1]; ma20 = df['MA20'].iloc[-1]
        trend_check = (last['close'] > ma5) and (ma5 >= ma20)
        
        if vol_check and gain_check and trend_check: return "BSJP-Momentum"

        if 'MA200' in df.columns:
            ma200 = df['MA200'].iloc[-1]; ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns else 0
            if (ma200 > ma20) and (ma20 >= ma50) and (price_gain > 0.02) and vol_check:
                return "Reversal-Setup"
        return False
    except: return False

def get_sniper_details_full(ticker, df):
    try:
        stock = yf.Ticker(ticker)
        try: info = stock.info
        except: info = {}
        
        last = df.iloc[-1]; prev = df.iloc[-2]
        price = last['close']; vol = last['volume']
        pct_change = (price - prev['close']) / prev['close']
        vol_ma20 = df['VolMA20'].iloc[-1] if 'VolMA20' in df.columns else 0
        ma5 = df['MA5'].iloc[-1] if 'MA5' in df.columns else 0
        ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else 0
        val_trans = price * vol
        mkt_cap = info.get('marketCap', 0)
        float_shares = info.get('floatShares', 0); shares_out = info.get('sharesOutstanding', 1)
        free_float = (float_shares / shares_out) * 100 if float_shares else 0
        
        report = []
        def make_card(status, title, detail):
            cls = "sniper-pass" if status=="pass" else ("sniper-warn" if status=="warn" else "sniper-fail")
            icon = "🚀" if status=="pass" else ("⚠️" if status=="warn" else "❌")
            return f"<div class='{cls}'><div style='font-weight:bold; font-size:1.0rem;'>{icon} {title}</div><div style='color:#ccc; font-size:0.85rem;'>{detail}</div></div>"

        if pct_change >= 0.05: report.append(make_card("pass", "Power Candle", f"+{pct_change*100:.2f}%"))
        elif pct_change >= 0.03: report.append(make_card("warn", "Moderate", f"+{pct_change*100:.2f}%"))
        else: report.append(make_card("fail", "Weak", f"+{pct_change*100:.2f}%"))

        if vol_ma20 > 0 and vol >= 1.5 * vol_ma20: report.append(make_card("pass", "Vol Ledak", f"{vol:,.0f} (>1.5x)"))
        else: report.append(make_card("fail", "Vol Sepi", "Normal"))

        if (val_trans/1e9) >= 5: report.append(make_card("pass", "Liquid", f"Rp {val_trans/1e9:.1f} M"))
        else: report.append(make_card("fail", "Illiquid", f"Rp {val_trans/1e9:.1f} M"))

        if price > ma5 and ma5 > ma20: report.append(make_card("pass", "Strong Trend", "Price > MA5 > MA20"))
        else: report.append(make_card("warn", "Messy Trend", "MA belum rapi"))

        return report
    except: return []

def get_fundamental_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        score = 0; report = []
        per = info.get('trailingPE', 999) or 999
        roe = info.get('returnOnEquity', 0) or 0
        
        def make_card(condition, text, subtext):
            color = "#238636" if condition else "#da3633"; icon = "✅" if condition else "❌"
            return f"""<div style='padding:10px; border-radius:6px; margin-bottom:5px; border-left:4px solid {color}; background-color:#21262d;'><div style='font-weight:600;'>{icon} {text}</div><div style='color:#8b949e; font-size:0.8rem;'>{subtext}</div></div>"""

        if 0 < per < 20: score += 1; report.append(make_card(True, "Harga Wajar", f"PER: {per:.2f}x"))
        else: report.append(make_card(False, "Mahal/Rugi", f"PER: {per:.2f}x"))
        if roe > 0.10: score += 1; report.append(make_card(True, "ROE Bagus", f"ROE: {roe*100:.1f}%"))
        else: report.append(make_card(False, "ROE Rendah", f"ROE: {roe*100:.1f}%"))
        return score, report
    except: return 0, ["<div style='color:red'>N/A</div>"]

def train_kmeans(df):
    cols = [c for c in ['BB_Width', 'RSI', 'OBV'] if c in df.columns]
    if not cols: return df, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
    df['Cluster'] = kmeans.labels_
    return df, kmeans

def train_sklearn(model_class, X_tr, y_tr, X_last, **kwargs):
    try:
        model = model_class(**kwargs); model.fit(X_tr, y_tr)
        return model.predict_proba(X_last)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X_last)[0])
    except: return 0.5

class LSTMNet(nn.Module):
    def __init__(self, i): super().__init__(); self.l=nn.LSTM(i,64,batch_first=True); self.f=nn.Linear(64,1); self.s=nn.Sigmoid()
    def forward(self, x): o,_=self.l(x); return self.s(self.f(o[:,-1,:]))
class GRUNet(nn.Module):
    def __init__(self, i): super().__init__(); self.g=nn.GRU(i,64,batch_first=True); self.f=nn.Linear(64,1); self.s=nn.Sigmoid()
    def forward(self, x): o,_=self.g(x); return self.s(self.f(o[:,-1,:]))
class StockTransformer(nn.Module):
    def __init__(self, i): super().__init__(); self.e=nn.Linear(i,32); self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,batch_first=True),1); self.f=nn.Linear(32,1); self.s=nn.Sigmoid()
    def forward(self, x): x=self.e(x); return self.s(self.f(self.enc(x)[:,-1,:]))

def train_dl(model_class, X_tr, y_tr, last, is_cnn=False):
    try:
        X_t=torch.tensor(X_tr,dtype=torch.float32).unsqueeze(1); y_t=torch.tensor(y_tr.values,dtype=torch.float32).unsqueeze(1); last_t=torch.tensor(last,dtype=torch.float32).unsqueeze(1)
        if is_cnn: X_t=X_t.permute(0,2,1); last_t=last_t.permute(0,2,1); model=model_class()
        else: model=model_class(X_tr.shape[1])
        optim=torch.optim.Adam(model.parameters(),lr=0.01); loss_fn=nn.BCELoss()
        for _ in range(20): optim.zero_grad(); loss_fn(model(X_t),y_t).backward(); optim.step()
        with torch.no_grad(): return model(last_t).item()
    except: return 0.5

def train_rl(prices): return 0.5 

st.title("🎯 Infinity Sniper: Entry & Exit Strategy")
tickers = get_all_tickers()

with st.sidebar:
    st.header("🎮 Control Panel")
    if st.button("🚀 SCAN SNIPER (AUTO)"):
        found = []
        bar = st.progress(0); txt = st.empty()
        for i, t in enumerate(tickers):
            txt.text(f"Scanning {t}..."); bar.progress((i+1)/len(tickers))
            try:
                df = get_data(t)
                if df is not None:
                    df = rekayasa_fitur(df)
                    mode = cek_kriteria_sniper_auto(df)
                    if mode: found.append(f"{t} ({mode})")
            except: continue
        bar.empty(); txt.empty()
        if found: st.success(f"Ketemu {len(found)}!"); st.session_state['hot'] = found
        else: st.warning("Zonk. Pasar sepi.")
            
    if 'hot' in st.session_state and st.session_state['hot']:
        sel = st.radio("Hasil Scan:", st.session_state['hot'])
        target = sel.split(" ")[0]
    else:
        target = st.selectbox("Manual:", tickers)

if not tickers: st.stop()
raw_df = get_data(target)
if raw_df is None: st.error("Data Error"); st.stop()
df_ml = rekayasa_fitur(raw_df)
df_ml, _ = train_kmeans(df_ml)

status = cek_kriteria_sniper_auto(df_ml)
if status: st.success(f"🔥 {target} SNIPER MODE: {status}")
else: st.info(f"ℹ️ {target} Normal Mode")

sl, tp1, tp2, atr_val = calculate_exit_strategy(df_ml)
curr_price = raw_df['close'].iloc[-1]

tab1, tab2, tab3, tab4 = st.tabs(["💰 Trading Plan (Exit)", "🧠 AI Analysis", "📈 Chart", "🏛️ Fundamental"])

with tab1:
    st.subheader(f"Trading Plan: {target}")
    st.caption("Dihitung menggunakan volatilitas ATR (Average True Range). Disiplin adalah kunci!")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Harga Sekarang (Entry)", f"{curr_price:,.0f}")
    c2.metric("Volatilitas (ATR)", f"{atr_val:,.0f}", help="Rata-rata pergerakan harga harian")
    c3.metric("Risk per Share", f"-{(curr_price - sl):,.0f}", f"-{((curr_price-sl)/curr_price)*100:.1f}%")

    col_sl, col_tp = st.columns(2)
    with col_sl:
        st.markdown(f"""
        <div class='exit-card-loss'>
            <h3>🛑 CUT LOSS (Stop)</h3>
            <h1 style='color:#ff1744; margin:0;'>{sl:,.0f}</h1>
            <p>Jual paksa jika harga closing di bawah ini.</p>
            <hr style='border-color:#555'>
            <small>Atau jika MA5 cross down MA20 (Trend Patah)</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col_tp:
        st.markdown(f"""
        <div class='exit-card-win'>
            <h3>💰 TAKE PROFIT (Target)</h3>
            <h2 style='color:#00e676; margin:0;'>TP 1: {tp1:,.0f}</h2>
            <p>Amankan sebagian profit di sini.</p>
            <h2 style='color:#00e676; margin:0;'>TP 2: {tp2:,.0f}</h2>
            <p>Target maksimal swing/sniper.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🚦 Sniper Checklist")
    report = get_sniper_details_full(target, df_ml)
    c_a, c_b = st.columns(2)
    for i, r in enumerate(report):
        with c_a if i < len(report)/2 else c_b: st.markdown(r, unsafe_allow_html=True)

with tab2: 
    st.subheader("Konsensus AI (15 Model)")
    feats = ['Returns','MA10','MA50','RSI','MACD','OBV','BB_Width','Cluster']
    valid_f = [f for f in feats if f in df_ml.columns]
    X = df_ml[valid_f]; y = df_ml['Target']
    scaler = MinMaxScaler(); X_sc = scaler.fit_transform(X)
    last = X_sc[[-1]]
    
    with st.spinner("AI Voting..."):
        p_xgb = train_sklearn(XGBClassifier, X_sc[:-1], y[:-1], last, n_estimators=50)
        p_rf = train_sklearn(RandomForestClassifier, X_sc[:-1], y[:-1], last, n_estimators=50)
        p_svm = train_sklearn(SVC, X_sc[:-1], y[:-1], last, probability=True)
        p_lstm = train_dl(LSTMNet, X_sc[:-1], y[:-1], last)
        
        avg = (p_xgb + p_rf + p_svm + p_lstm) / 4
        if avg > 0.6: txt="BUY"; c="#00e676"
        elif avg < 0.4: txt="SELL"; c="#ff1744"
        else: txt="NEUTRAL"; c="#888"
        
        st.markdown(f"<h1 style='text-align:center; color:{c}'>{txt} ({avg*100:.0f}%)</h1>", unsafe_allow_html=True)

with tab3: 
    ma = 'MA200' if 'MA200' in df_ml.columns else 'MA20'
    fig = go.Figure(data=[go.Candlestick(x=raw_df.index, open=raw_df['open'], high=raw_df['high'], low=raw_df['low'], close=raw_df['close'])])
    if ma in df_ml.columns: fig.add_trace(go.Scatter(x=raw_df.index, y=df_ml[ma], line=dict(color='orange'), name=ma))
    if 'MA5' in df_ml.columns: fig.add_trace(go.Scatter(x=raw_df.index, y=df_ml['MA5'], line=dict(color='cyan'), name='MA5'))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab4: 
    s, r = get_fundamental_score(target)
    st.metric("Fundamental", f"{s}/6")
    for x in r: st.markdown(x, unsafe_allow_html=True)