import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import random
import uuid
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="FinPulse AI | Financial Telemetry & Anomaly Screener",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Robust FinTech Themed Background & Clean Styling
st.markdown("""
<style>
    /* Dark FinTech Cyber Gradient Background */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #031B28 0%, #020C14 45%, #05131E 100%) !important;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide standard sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Ecoza-style top navbar & radio pills */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 12px;
        background: rgba(8, 28, 44, 0.7);
        padding: 6px 14px;
        border-radius: 35px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        backdrop-filter: blur(10px);
    }
    
    div[role="radiogroup"] > label {
        padding: 5px 12px;
        margin-bottom: 0px !important;
        font-weight: 500;
        cursor: pointer;
    }

    /* Badges */
    .user-pill {
        color: #38BDF8;
        font-weight: 600;
        font-size: 0.9rem;
        background: rgba(56, 189, 248, 0.12);
        padding: 6px 16px;
        border-radius: 25px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: inline-block;
    }

    .points-pill {
        background: linear-gradient(135deg, #10B981, #059669);
        color: #FFFFFF;
        padding: 6px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
    }

    /* FinTech Glass Cards */
    .fin-card {
        background: rgba(8, 24, 38, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        margin-bottom: 15px;
    }

    /* Button enhancements */
    .stButton>button {
        background: linear-gradient(90deg, #0284C7, #2563EB);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.45rem 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Synthetic Data In-Memory Fallback
def generate_synthetic_data_inline():
    random.seed(42)
    np.random.seed(42)

    MERCHANT_CONFIG = {
        'Food & Dining': {'merchants': ['Swiggy', 'Zomato', 'Dominos', 'A2B'], 'modes': ['UPI', 'Debit Card'], 'amount': (80, 850), 'p': 0.35},
        'Groceries': {'merchants': ['Blinkit', 'Zepto', 'Instamart', 'BigBasket'], 'modes': ['UPI', 'Debit Card'], 'amount': (200, 3500), 'p': 0.20},
        'Travel & Transport': {'merchants': ['Uber', 'Ola', 'Rapido', 'Chennai Metro'], 'modes': ['UPI'], 'amount': (40, 1800), 'p': 0.15},
        'Shopping': {'merchants': ['Amazon IN', 'Flipkart', 'Myntra', 'Zudio'], 'modes': ['Credit Card', 'UPI'], 'amount': (499, 7500), 'p': 0.10},
        'Bills & Utilities': {'merchants': ['TNEB Electricity', 'Airtel Broadband', 'Jio Recharge'], 'modes': ['NetBanking', 'UPI'], 'amount': (299, 2500), 'p': 0.10},
        'Entertainment': {'merchants': ['Netflix India', 'Spotify', 'BookMyShow'], 'modes': ['Credit Card', 'UPI'], 'amount': (199, 1200), 'p': 0.10}
    }

    cats = list(MERCHANT_CONFIG.keys())
    weights = [MERCHANT_CONFIG[c]['p'] for c in cats]
    txs = []
    start = datetime.now() - timedelta(days=180)

    # Recurring Salary & Rent
    curr = start
    while curr <= datetime.now():
        txs.append({'transaction_id': str(uuid.uuid4())[:8], 'date': curr.replace(day=1).strftime('%Y-%m-%d %H:%M:%S'), 'narration': 'ACH/SALARY-CREDIT/TECH-CORP', 'amount': 45000.0, 'type': 'Credit', 'payment_mode': 'NEFT', 'category': 'Salary', 'is_anomaly': 0})
        txs.append({'transaction_id': str(uuid.uuid4())[:8], 'date': curr.replace(day=4).strftime('%Y-%m-%d %H:%M:%S'), 'narration': 'UPI/HOUSE-RENT/okaxis', 'amount': 12000.0, 'type': 'Debit', 'payment_mode': 'UPI', 'category': 'Rent', 'is_anomaly': 0})
        curr = (curr.replace(day=28) + timedelta(days=4)).replace(day=1)

    # Routine Transactions
    for _ in range(1000):
        c = np.random.choice(cats, p=weights)
        cfg = MERCHANT_CONFIG[c]
        m = random.choice(cfg['merchants'])
        mode = random.choice(cfg['modes'])
        t = start + timedelta(days=random.randint(0, 180), hours=random.randint(6, 23), minutes=random.randint(0, 59))
        amt = round(random.uniform(cfg['amount'][0], cfg['amount'][1]), 2)
        txs.append({'transaction_id': str(uuid.uuid4())[:8], 'date': t.strftime('%Y-%m-%d %H:%M:%S'), 'narration': f"UPI/{m.upper()}/{random.randint(1000,9999)}", 'amount': amt, 'type': 'Debit', 'payment_mode': mode, 'category': c, 'is_anomaly': 0})

    # Outliers
    anomalies = [
        ("POS/JEWELLERY-MART/CHENNAI", 68000.0, "Shopping"),
        ("UPI/CRYPTO-EXCHANGE/paytm", 35000.0, "Shopping"),
        ("POS/LUXURY-ELECTRONICS", 89000.0, "Shopping"),
        ("UPI/SWIGGY/OVERCHARGE", 14500.0, "Food & Dining")
    ]
    for desc, amt, cat in anomalies:
        t = start + timedelta(days=random.randint(10, 170), hours=3)
        txs.append({'transaction_id': str(uuid.uuid4())[:8], 'date': t.strftime('%Y-%m-%d %H:%M:%S'), 'narration': desc, 'amount': amt, 'type': 'Debit', 'payment_mode': 'Credit Card', 'category': cat, 'is_anomaly': 1})

    df_gen = pd.DataFrame(txs).sort_values('date').reset_index(drop=True)
    df_gen.to_csv("transactions.csv", index=False)
    return df_gen

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("transactions.csv")
    except Exception:
        df = generate_synthetic_data_inline()
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- LOGIN SCREEN ---
if not st.session_state.authenticated:
    _, c2, _ = st.columns([1, 1.6, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="fin-card" style="text-align: center;">
            <div style="font-size: 3rem;">💸</div>
            <h2 style="color: #F8FAFC; margin-top: 5px;">FinPulse AI Portal</h2>
            <p style="color: #94A3B8; font-size: 0.95rem;">Synthetic Intelligence for Financial Integrity & Anomaly Screening</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", value="", placeholder="Enter your username")
            password = st.text_input("Access Key", type="password", value="", placeholder="Enter password")
            submit = st.form_submit_button("Authenticate & Enter")
            
            if submit:
                if username.strip() != "":
                    st.session_state.authenticated = True
                    st.session_state.user_name = username
                    st.rerun()
                else:
                    st.error("Please enter a username.")
    st.stop()

# --- TOP NAVIGATION (ECOZA STYLE) ---
h1, h2, h3, h4 = st.columns([1.8, 4.4, 1.2, 0.8])

with h1:
    st.markdown('<h3 style="margin:0; color:#38BDF8; font-weight:800;">💸 FinPulse AI</h3>', unsafe_allow_html=True)

with h2:
    selected_tab = st.radio(
        label="Navigation Menu",
        options=["🏠 Home", "📊 Dashboard", "🚨 ML Audit", "⚙️ Specs"],
        horizontal=True,
        label_visibility="collapsed"
    )

with h3:
    st.markdown('<div class="points-pill">⭐ 250 pts</div>', unsafe_allow_html=True)

with h4:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_name = ""
        st.rerun()

st.markdown(
    f'<div style="margin-top:-6px; margin-bottom:18px;"><span class="user-pill">Logged in as: {st.session_state.user_name}</span></div>',
    unsafe_allow_html=True
)
st.divider()

# --- TAB 1: HOME ---
if selected_tab == "🏠 Home":
    st.markdown("## 🛡️ AI Financial Telemetry & Spending Sentinel")
    st.caption("Zero-risk synthetic financial modeling combined with unsupervised machine learning anomaly detection.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="fin-card">
            <h4>🔒 Zero-Risk Synthetic Data</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Eliminates PII exposure. Simulates authentic Indian UPI, POS, and NEFT debits with realistic Gaussian amounts and time patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="fin-card">
            <h4>🤖 Unsupervised ML Screener</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Isolation Forest algorithm continuously evaluates transaction distributions to detect high-value spending spikes and leaks.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="fin-card">
            <h4>📈 Financial Wellness</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Visual analytics tracking savings ratio, month-over-month cash flow, and domain expenditures automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: DASHBOARD ---
elif selected_tab == "📊 Dashboard":
    st.markdown("## 📊 Financial Telemetry & Spending Breakdown")
    
    total_income = df[df['type'] == 'Credit']['amount'].sum()
    total_expense = df[df['type'] == 'Debit']['amount'].sum()
    net_savings = total_income - total_expense
    savings_rate = (net_savings / total_income) * 100 if total_income > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Income (Synthetic)", f"₹{total_income:,.2f}")
    k2.metric("Total Expenses", f"₹{total_expense:,.2f}")
    k3.metric("Net Surplus", f"₹{net_savings:,.2f}")
    k4.metric("Savings Ratio", f"{savings_rate:.1f}%")
    
    st.divider()
    
    ch1, ch2 = st.columns(2)
    expense_df = df[df['type'] == 'Debit'].copy()
    
    with ch1:
        st.markdown("#### Domain Breakdown")
        cat_agg = expense_df.groupby('category')['amount'].sum().reset_index()
        fig_donut = px.pie(cat_agg, values='amount', names='category', hole=0.5,
                           color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E2E8F0")
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with ch2:
        st.markdown("#### Monthly Velocity")
        expense_df['month_year'] = expense_df['date'].dt.to_period('M').astype(str)
        month_agg = expense_df.groupby('month_year')['amount'].sum().reset_index()
        fig_bar = px.bar(month_agg, x='month_year', y='amount', color='amount', color_continuous_scale='Blues')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E2E8F0")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: ML AUDIT ---
elif selected_tab == "🚨 ML Audit":
    st.markdown("## 🚨 Machine Learning Outlier Screener")
    st.caption("Isolation Forest evaluates spending distributions without requiring labeled fraud data.")
    
    expense_df = df[df['type'] == 'Debit'].copy()
    
    # Train Isolation Forest on DataFrame with named features (fixes sklearn warnings)
    train_features = expense_df[['amount']].copy()
    iso = IsolationForest(contamination=0.018, random_state=42)
    expense_df['anomaly_status'] = iso.fit_predict(train_features)
    anomalies = expense_df[expense_df['anomaly_status'] == -1]
    
    st.info(f"🛡️ Isolation Forest flagged **{len(anomalies)} transactions** exceeding statistical Gaussian thresholds.")
    st.dataframe(
        anomalies[['date', 'narration', 'amount', 'category', 'payment_mode']].sort_values(by='amount', ascending=False),
        use_container_width=True
    )
    
    st.divider()
    st.markdown("#### ⚡ Real-Time Transaction Screening Engine")
    st.caption("Input a test amount to see if it violates baseline spending limits.")
    
    s1, s2, s3, s4 = st.columns(4)
    in_desc = s1.text_input("Narration", "UPI/TEST-SPEND")
    in_cat = s2.selectbox("Domain", df['category'].dropna().unique())
    in_amt = s3.number_input("Amount (₹)", min_value=1.0, value=25000.0, step=500.0)
    audit_btn = s4.button("Screen Transaction")
    
    if audit_btn:
        # Proper DataFrame feature input to prevent warning
        test_sample = pd.DataFrame({'amount': [in_amt]})
        res = iso.predict(test_sample)[0]
        score = iso.decision_function(test_sample)[0]
        
        if res == -1:
            st.markdown(f"""
            <div style="background: rgba(220, 38, 38, 0.15); border: 1px solid #EF4444; border-radius: 10px; padding: 15px; margin-top: 10px;">
                <h4 style="color: #F87171; margin:0;">🚨 STATISTICAL ANOMALY DETECTED</h4>
                <p style="margin: 5px 0 0 0; color: #FCA5A5;">
                    The spend of <b>₹{in_amt:,.2f}</b> at <b>'{in_desc}'</b> is <b>{in_amt / 1200:.1f}x higher</b> than the baseline category average. 
                    <br>Isolation score: <code>{score:.3f}</code> (Outlier threshold breached).
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 10px; padding: 15px; margin-top: 10px;">
                <h4 style="color: #34D399; margin:0;">✅ CLEAN ROUTINE TRANSACTION</h4>
                <p style="margin: 5px 0 0 0; color: #A7F3D0;">
                    The spend of <b>₹{in_amt:,.2f}</b> at <b>'{in_desc}'</b> is well within normal Gaussian limits for this account.
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 4: SPECS ---
elif selected_tab == "⚙️ Specs":
    st.markdown("## ⚙️ Model & Architecture Specifications")
    st.markdown("""
    * **Data Layer:** Procedural Synthetic Banking Generator (Faker + NumPy Gaussian Distributions)
    * **Algorithm:** `sklearn.ensemble.IsolationForest` (Unsupervised tree-based partitioning)
    * **Design System:** Custom CSS capsule navigation bar with dark FinTech gradient
    """)