import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG & GIAO DIỆN MƯỢT MÀ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ames, Iowa Home Advisor - Định Giá & Chọn Mua Nhà Tại Mỹ",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 1.95rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

SQFT_TO_M2 = 0.092903

# ---------------------------------------------------------
# 2. MÔ HÌNH RỪNG NGẪU NHIÊN THUẦN NUMPY (NHANH, 0 LỖI THƯ VIỆN)
# ---------------------------------------------------------
class FastDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=8, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(42)
        self.tree = None

    def _fit(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) < 1e-6:
            return float(np.mean(y))

        feat_indices = self.rng.choice(n_features, size=self.max_features or n_features, replace=False)
        best_gain = -1.0
        best_feat, best_thresh = None, None
        parent_var = np.var(y) * n_samples

        for f in feat_indices:
            vals = X[:, f]
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            thresholds = np.percentile(vals, np.linspace(12, 88, min(14, len(uniq))))
            for t in np.unique(thresholds):
                left_mask = vals <= t
                n_l = int(np.sum(left_mask))
                n_r = n_samples - n_l
                if n_l < 3 or n_r < 3:
                    continue
                gain = parent_var - (np.var(y[left_mask]) * n_l + np.var(y[~left_mask]) * n_r)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = float(t)

        if best_feat is None:
            return float(np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        return (
            best_feat,
            best_thresh,
            self._fit(X[left_mask], y[left_mask], depth + 1),
            self._fit(X[~left_mask], y[~left_mask], depth + 1)
        )

    def fit(self, X, y):
        self.tree = self._fit(X, y, 0)
        return self

    def _pred_one(self, row, node):
        if not isinstance(node, tuple):
            return node
        feat, thresh, left, right = node
        return self._pred_one(row, left if row[feat] <= thresh else right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._pred_one(row, self.tree) for row in X])


class FastRandomForest:
    def __init__(self, n_estimators=45, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
        self.estimators_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n_samples, size=n_samples)
            tree = FastDecisionTree(
                max_depth=self.max_depth,
                max_features=max(1, int(n_features * 0.75)),
                rng=self.rng
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

        sub_idx = self.rng.choice(n_samples, size=min(300, n_samples), replace=False)
        X_sub, y_sub = X[sub_idx], y[sub_idx]
        base_mse = np.mean((y_sub - self.predict(X_sub)) ** 2)
        imps = np.zeros(n_features)
        for f in range(n_features):
            X_perm = X_sub.copy()
            X_perm[:, f] = self.rng.permutation(X_perm[:, f])
            perm_mse = np.mean((y_sub - self.predict(X_perm)) ** 2)
            imps[f] = max(0.0, perm_mse - base_mse)
        self.feature_importances_ = imps / imps.sum() if imps.sum() > 0 else np.ones(n_features) / n_features
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        all_preds = np.array([t.predict(X) for t in self.estimators_])
        return np.mean(all_preds, axis=0)


# ---------------------------------------------------------
# 3. DANH MỤC PHÂN KHU TẠI AMES, IOWA & DỮ LIỆU THỊ TRƯỜNG
# ---------------------------------------------------------
AMES_NEIGHBORHOODS = {
    "NoRidge / NridgHt (Northridge - Khu biệt thự cao cấp Bắc Ames)": {"tier": 5, "code": "NoRidge", "desc": "Gần hồ Ada Hayden, nhà lớn, sang trọng nhất Ames"},
    "StoneBr (Stone Brook - Khu dân cư cao cấp Đông Bắc)": {"tier": 5, "code": "StoneBr", "desc": "Yên tĩnh, mật độ thấp, thiết kế hiện đại"},
    "Somerst (Somerset - Khu đô thị hiện đại & tiện ích)": {"tier": 4, "code": "Somerst", "desc": "Quy hoạch đồng bộ, nhà hàng & công viên nội khu"},
    "CollgCr (College Creek - Gần ĐH Bang Iowa / ISU)": {"tier": 3, "code": "CollgCr", "desc": "Phía Tây Nam Ames, rất được ưa chuộng bởi giảng viên & gia đình"},
    "Gilbert / Timber (Khu gia đình trẻ & trường học tốt)": {"tier": 3, "code": "Gilbert", "desc": "Không gian xanh, phù hợp gia đình có con nhỏ"},
    "NWAmes / SawyerW (Tây Bắc Ames - Dân cư ổn định)": {"tier": 2, "code": "NWAmes", "desc": "Khu trung lưu lâu đời, sân vườn rộng, giao thông thuận tiện"},
    "OldTown / Edwards (Trung tâm phố cổ Ames - Giá mềm)": {"tier": 1, "code": "OldTown", "desc": "Gần trung tâm & campus ISU, giá hợp lý, tiềm năng cải tạo/cho thuê"}
}

CODE_TO_TIER = {
    "NoRidge": 5, "NridgHt": 5, "StoneBr": 5,
    "Somerst": 4, "Timber": 4, "Veenker": 4,
    "CollgCr": 3, "Gilbert": 3, "ClearCr": 3, "Crawfor": 3, "Blmngtn": 3,
    "NWAmes": 2, "SawyerW": 2, "Mitchel": 2, "NAmes": 2, "NPkVill": 2, "SWISU": 2,
    "OldTown": 1, "Edwards": 1, "Sawyer": 1, "BrkSide": 1, "IDOTRR": 1, "MeadowV": 1, "BrDale": 1
}

TIER_TO_LABEL = {
    5: "Northridge / Stone Brook (Cao cấp)",
    4: "Somerset / Timberland (Hiện đại)",
    3: "College Creek / Gilbert (Gần ISU)",
    2: "NW Ames / Sawyer West (Trung lưu)",
    1: "Old Town / Edwards (Phố cổ - Giá mềm)"
}


@st.cache_data
def generate_ames_market_data(n_samples: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    gr_liv_area = np.round(rng.uniform(750, 3400, n_samples), 0)
    lot_area = np.round(np.clip(gr_liv_area * rng.uniform(3.2, 7.5, n_samples), 2500, 28000), 0)
    total_bsmt_sf = np.round(gr_liv_area * rng.uniform(0.45, 0.95, n_samples), 0)
    neighborhood_tier = rng.choice([5, 4, 3, 2, 1], size=n_samples, p=[0.14, 0.16, 0.28, 0.24, 0.18])
    overall_qual = np.clip(np.round(3.5 + neighborhood_tier * 0.85 + rng.normal(0, 1.1, n_samples)), 2, 10).astype(int)
    year_built = np.clip(np.round(1945 + neighborhood_tier * 11 + overall_qual * 2.2 + rng.normal(0, 12, n_samples)), 1920, 2026).astype(int)
    bedrooms = np.clip(np.round(gr_liv_area / 520 + rng.normal(0, 0.5, n_samples)), 1, 6).astype(int)
    full_bath = np.clip(np.round(gr_liv_area / 850 + rng.normal(0, 0.35, n_samples)), 1, 4).astype(int)
    garage_cars = np.clip(np.round(overall_qual / 3.3 + rng.normal(0, 0.45, n_samples)), 0, 4).astype(int)
    central_air = (rng.random(n_samples) < (0.65 + neighborhood_tier * 0.07)).astype(int)

    tier_mult = np.select(
        [neighborhood_tier == 5, neighborhood_tier == 4, neighborhood_tier == 3, neighborhood_tier == 2, neighborhood_tier == 1],
        [1.34, 1.18, 1.05, 0.94, 0.82]
    )
    qual_mult = 0.62 + (overall_qual ** 1.32) * 0.048
    age_factor = np.maximum(0.80, 1.0 - (2026 - year_built) * 0.0028)
    air_mult = np.where(central_air == 1, 1.05, 0.91)

    base_price = (
        gr_liv_area * 82.0
        + total_bsmt_sf * 34.0
        + lot_area * 1.45
        + garage_cars * 13500.0
        + full_bath * 7500.0
        + bedrooms * 3500.0
    )
    noise = rng.normal(1.0, 0.032, n_samples)
    sale_price = np.round(np.maximum(65000, base_price * tier_mult * qual_mult * age_factor * air_mult * noise), -2)

    return pd.DataFrame({
        "GrLivArea": gr_liv_area,
        "LotArea": lot_area,
        "TotalBsmtSF": total_bsmt_sf,
        "NeighborhoodTier": neighborhood_tier,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "GarageCars": garage_cars,
        "CentralAir": central_air,
        "SalePrice": sale_price
    })


@st.cache_resource
def train_ames_model(df: pd.DataFrame):
    feature_cols = [
        "GrLivArea", "LotArea", "TotalBsmtSF", "NeighborhoodTier",
        "OverallQual", "YearBuilt", "BedroomAbvGr", "FullBath",
        "GarageCars", "CentralAir"
    ]
    X = df[feature_cols].values
    y = df["SalePrice"].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = FastRandomForest(n_estimators=45, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = 1.0 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    mae = float(np.mean(np.abs(y_test - preds)))
    return model, feature_cols, float(r2), mae


def predict_with_interval(model, input_df: pd.DataFrame):
    tree_preds = np.array([tree.predict(input_df.values)[0] for tree in model.estimators_])
    return float(np.mean(tree_preds)), float(np.percentile(tree_preds, 15)), float(np.percentile(tree_preds, 85))


if "saved_ames_houses" not in st.session_state:
    st.session_state.saved_ames_houses = []
if "custom_ames_df" not in st.session_state:
    st.session_state.custom_ames_df = None

df_market = st.session_state.custom_ames_df if st.session_state.custom_ames_df is not None else generate_ames_market_data()
model, feature_cols, model_r2, model_mae = train_ames_model(df_market)

# ---------------------------------------------------------
# 4. THANH BÊN (SIDEBAR): THÔNG SỐ NHÀ TẠI AMES, IOWA (HOA KỲ)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🇺🇸 Cấu Hình Nhà Tại Ames, Iowa")
    st.caption("Nhập thông số theo chuẩn bất động sản Hoa Kỳ (tự động quy đổi m² & VNĐ):")

    ten_phuong_an = st.text_input(
        "🏷️ Tên gợi nhớ căn nhà",
        value=f"Căn #{len(st.session_state.saved_ames_houses) + 1} - Ames IA"
    )

    ngan_sach_usd = st.number_input(
        "💰 Ngân sách tối đa ($ USD)",
        min_value=50000, max_value=2000000, value=285000, step=5000
    )

    ty_gia_vnd = st.number_input(
        "💱 Tỷ giá quy đổi (VNĐ / 1 USD)",
        min_value=20000, max_value=30000, value=25400, step=100
    )

    st.divider()
    st.subheader("1. Khu vực & Diện tích (sq ft)")
    nb_label = st.selectbox(
        "📍 Phân khu (Neighborhood tại Ames, IA)",
        options=list(AMES_NEIGHBORHOODS.keys()),
        index=3
    )
    nb_info = AMES_NEIGHBORHOODS[nb_label]
    nb_tier = nb_info["tier"]
    st.caption(f"ℹ️ *{nb_info['desc']}*")

    gr_liv_area = st.number_input(
        "🏠 Diện tích sàn sinh hoạt (GrLivArea - sq ft)",
        min_value=500, max_value=5000, value=1650, step=50
    )
    st.caption(f"≈ **{gr_liv_area * SQFT_TO_M2:.1f} m²** diện tích ở trên mặt đất")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        total_bsmt_sf = st.number_input(
            "🧱 Tầng hầm (sq ft)",
            min_value=0, max_value=3500, value=1050, step=50
        )
    with col_a2:
        lot_area = st.number_input(
            "🌳 Khuôn viên đất (sq ft)",
            min_value=1500, max_value=50000, value=9200, step=200
        )
    st.caption(f"Tầng hầm ≈ **{total_bsmt_sf * SQFT_TO_M2:.1f} m²** | Khuôn viên ≈ **{lot_area * SQFT_TO_M2:.0f} m²**")

    st.subheader("2. Chất lượng & Công năng")
    overall_qual = st.slider(
        "⭐ Chất lượng xây dựng & hoàn thiện (1–10)",
        min_value=1, max_value=10, value=7
    )
    year_built = st.slider(
        "📅 Năm xây dựng (Year Built)",
        min_value=1920, max_value=2026, value=2004
    )

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        bedrooms = st.number_input("🛏️ Ngủ", min_value=1, max_value=8, value=3, step=1)
    with col_r2:
        full_bath = st.number_input("🚿 Tắm", min_value=1, max_value=5, value=2, step=1)
    with col_r3:
        garage_cars = st.selectbox("🚗 Gara", options=[0, 1, 2, 3, 4], index=2)

    central_air = st.checkbox("❄️ Có điều hòa & sưởi trung tâm (Central Air/HVAC)", value=True)

user_input = pd.DataFrame([{
    "GrLivArea": float(gr_liv_area),
    "LotArea": float(lot_area),
    "TotalBsmtSF": float(total_bsmt_sf),
    "NeighborhoodTier": int(nb_tier),
    "OverallQual": int(overall_qual),
    "YearBuilt": int(year_built),
    "BedroomAbvGr": int(bedrooms),
    "FullBath": int(full_bath),
    "GarageCars": int(garage_cars),
    "CentralAir": int(central_air)
}])

pred_usd, low_usd, high_usd = predict_with_interval(model, user_input)
closing_cost_usd = round(pred_usd * 0.025, 0)  # ~2.5% phí thủ tục sang tên/escrow tại Iowa
total_cash_needed_usd = pred_usd + closing_cost_usd
price_per_sqft = pred_usd / gr_liv_area
price_per_m2_usd = pred_usd / (gr_liv_area * SQFT_TO_M2)
pred_ty_vnd = (pred_usd * ty_gia_vnd) / 1e9
budget_diff_usd = ngan_sach_usd - total_cash_needed_usd

# ---------------------------------------------------------
# 5. GIAO DIỆN CHÍNH & 6 TAB ỨNG DỤNG CHO THỊ TRƯỜNG AMES, IA
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏡 Trợ Lý Định Giá & Tư Vấn Chọn Mua Nhà Tại Ames, Iowa (Hoa Kỳ)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Định giá chuẩn thị trường Ames, IA ($ USD & sq ft), quy đổi trực quan sang m² & VNĐ, tính vay Mortgage PITI (gồm thuế đất Story County) và checklist kiểm định nhà Mỹ.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Định Giá Ames, IA",
    "🔍 2. Lọc & Gợi Ý Nhà",
    "🏦 3. Tính Vay Mortgage (PITI)",
    "📋 4. Chấm Điểm & Checklist Mỹ",
    "⚖️ 5. So Sánh Các Căn",
    "📁 6. Nạp File Nhóm 5"
])

# =========================================================
# TAB 1: ĐỊNH GIÁ & PHÂN TÍCH ĐA YẾU TỐ TẠI AMES, IOWA
# =========================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"Giá nhà dự đoán ({gr_liv_area:,} sq ft)",
            value=f"${pred_usd:,.0f}",
            delta=f"≈ {pred_ty_vnd:.2f} Tỷ VNĐ (${low_usd:,.0f} - ${high_usd:,.0f})",
            delta_color="off"
        )
    with c2:
        st.metric(
            label="Tổng chi phí trọn gói (+2.5% Closing)",
            value=f"${total_cash_needed_usd:,.0f}",
            delta=f"Phí thủ tục/sang tên: ~${closing_cost_usd:,.0f}",
            delta_color="off"
        )
    with c3:
        st.metric(
            label=f"Đơn giá ({gr_liv_area * SQFT_TO_M2:.1f} m² sàn)",
            value=f"${price_per_sqft:,.1f} / sq ft",
            delta=f"≈ ${price_per_m2_usd:,.0f} / m² sàn ở",
            delta_color="off"
        )
    with c4:
        st.metric(
            label="Đối chiếu ngân sách trọn gói",
            value=f"${abs(budget_diff_usd):,.0f}",
            delta="Dư ngân sách (An toàn)" if budget_diff_usd >= 0 else "Vượt ngân sách dự kiến",
            delta_color="normal" if budget_diff_usd >= 0 else "inverse"
        )

    if budget_diff_usd >= 8000:
        st.success(
            f"✅ **Khả thi cao tại Ames, IA:** Với ngân sách **${ngan_sach_usd:,.0f}**, căn nhà tại phân khu *{nb_info['code']}* "
            f"(**{gr_liv_area:,} sq ft ≈ {gr_liv_area * SQFT_TO_M2:.1f} m²**, tầng hầm **{total_bsmt_sf:,} sq ft**, gara **{garage_cars} xe**, "
            f"chất lượng **{overall_qual}/10**) hoàn toàn nằm trong tầm tay, còn dư khoảng **${budget_diff_usd:,.0f}** cho chi phí sửa sang hoặc dự phòng."
        )
    elif budget_diff_usd >= 0:
        st.info(
            f"ℹ️ **Sát ngân sách:** Tổng chi phí cả phí thủ tục (**${total_cash_needed_usd:,.0f}**) vừa khít ngân sách (**${ngan_sach_usd:,.0f}**). "
            "Tại thị trường Ames, IA, bạn có thể thương lượng người bán hỗ trợ một phần *Seller Concessions* (phí sang tên) hoặc xem phương án vay ở **Tab 3**."
        )
    else:
        st.warning(
            f"⚠️ **Vượt ngân sách ${abs(budget_diff_usd):,.0f}:** Để mua tại Ames trong tầm giá **${ngan_sach_usd:,.0f}**, bạn có thể cân nhắc "
            "chọn phân khu lân cận như *College Creek / NW Ames / Old Town*, giảm bớt diện tích sàn hoặc xem biểu đồ bên dưới để biết diện tích tối đa mua được."
        )

    col_btn1, _ = st.columns([1.5, 3])
    with col_btn1:
        if st.button("➕ Lưu căn nhà này vào bảng so sánh", use_container_width=True):
            ten_hien_thi = ten_phuong_an.strip() or f"Căn #{len(st.session_state.saved_ames_houses) + 1}"
            st.session_state.saved_ames_houses.append({
                "Tên phương án": ten_hien_thi,
                "Phân khu": nb_info["code"],
                "Sàn ở (sq ft)": int(gr_liv_area),
                "Quy đổi (m²)": round(gr_liv_area * SQFT_TO_M2, 1),
                "Tầng hầm (sq ft)": int(total_bsmt_sf),
                "Chất lượng": f"{overall_qual}/10",
                "Năm XD": int(year_built),
                "Gara": f"{garage_cars} xe",
                "Giá dự đoán ($)": round(pred_usd, 0),
                "Trọn gói ($)": round(total_cash_needed_usd, 0),
                "Đơn giá ($/sq ft)": round(price_per_sqft, 1),
                "Dư/Thiếu NS ($)": round(budget_diff_usd, 0)
            })
            st.toast(f"Đã lưu '{ten_hien_thi}' sang Tab 5!", icon="✅")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("📈 Biến thiên giá theo Diện tích (sq ft)")
        st.caption(f"Khảo sát mức giá tại phân khu {nb_info['code']} từ 800 sq ft (~74 m²) đến 3,200 sq ft (~297 m²):")

        sqft_grid = np.arange(800, 3250, 100)
        sim_df = pd.concat([user_input] * len(sqft_grid), ignore_index=True)
        sim_df["GrLivArea"] = sqft_grid
        sim_prices = model.predict(sim_df)

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=sqft_grid, y=sim_prices,
            mode="lines", name="Giá nhà dự đoán ($)",
            line=dict(color="#2563EB", width=3)
        ))
        fig_area.add_trace(go.Scatter(
            x=[gr_liv_area], y=[pred_usd],
            mode="markers+text", name="Căn đang chọn",
            text=[f"{gr_liv_area:,} sq ft (${pred_usd:,.0f})"],
            textposition="top left",
            marker=dict(color="#DC2626", size=12, symbol="diamond")
        ))
        fig_area.add_hline(
            y=ngan_sach_usd, line_dash="dash", line_color="#16A34A",
            annotation_text=f"Ngân sách: ${ngan_sach_usd:,.0f}",
            annotation_position="bottom right"
        )
        fig_area.update_layout(
            xaxis_title="Diện tích sàn sinh hoạt (sq ft)",
            yaxis_title="Giá dự đoán ($ USD)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        affordable_sqft = sqft_grid[sim_prices <= ngan_sach_usd]
        if len(affordable_sqft) > 0:
            max_sf = int(affordable_sqft.max())
            st.caption(f"💡 Với cấu hình hiện tại và ngân sách **${ngan_sach_usd:,.0f}**, diện tích tối đa bạn có thể chọn là khoảng **{max_sf:,} sq ft** (≈ **{max_sf * SQFT_TO_M2:.1f} m²**).")
        else:
            st.caption(f"💡 Với cấu hình hiện tại, mức 800 sq ft cũng vượt ngân sách **${ngan_sach_usd:,.0f}**.")

    with chart_col2:
        st.subheader("🧩 Mức độ ảnh hưởng của 10 đặc trưng Ames Housing")
        st.caption(f"Độ chính xác mô hình trên dữ liệu kiểm thử: R² = {model_r2*100:.1f}% (Sai số TB ±${model_mae:,.0f}):")

        label_vn = {
            "GrLivArea": "Diện tích sàn ở (GrLivArea)",
            "LotArea": "Diện tích khuôn viên (LotArea)",
            "TotalBsmtSF": "Diện tích tầng hầm (Basement)",
            "NeighborhoodTier": "Phân khu tại Ames (Neighborhood)",
            "OverallQual": "Chất lượng tổng thể (1–10)",
            "YearBuilt": "Năm xây dựng (YearBuilt)",
            "BedroomAbvGr": "Số phòng ngủ",
            "FullBath": "Số phòng tắm đầy đủ",
            "GarageCars": "Sức chứa Gara (GarageCars)",
            "CentralAir": "Điều hòa & sưởi trung tâm"
        }
        imp_df = pd.DataFrame({
            "Yếu tố": [label_vn[c] for c in feature_cols],
            "Tầm quan trọng (%)": model.feature_importances_ * 100
        }).sort_values("Tầm quan trọng (%)", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Tầm quan trọng (%)"],
            y=imp_df["Yếu tố"],
            orientation="h",
            text=imp_df["Tầm quan trọng (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker_color="#2563EB"
        ))
        fig_imp.update_layout(
            height=380,
            margin=dict(l=20, r=45, t=30, b=20),
            xaxis_title="Tỷ trọng tác động (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("🔄 Bảng Giả Định: Nếu Điều Chỉnh 1 Tiêu Chí Tại Ames Thì Giá Thay Đổi Bao Nhiêu?")
    scenarios = []
    s_plus200 = user_input.copy(); s_plus200["GrLivArea"] += 200
    p_plus200 = model.predict(s_plus200)[0]
    scenarios.append(("Mở rộng thêm +200 sq ft (~18.6 m²) sàn ở", f"{gr_liv_area + 200:,} sq ft", p_plus200, p_plus200 - pred_usd))

    if overall_qual < 10:
        s_q = user_input.copy(); s_q["OverallQual"] = overall_qual + 1
        p_q = model.predict(s_q)[0]
        scenarios.append(("Nâng cấp chất lượng hoàn thiện +1 bậc", f"{overall_qual + 1}/10 điểm", p_q, p_q - pred_usd))

    if nb_tier > 1:
        s_nb = user_input.copy(); s_nb["NeighborhoodTier"] = nb_tier - 1
        p_nb = model.predict(s_nb)[0]
        scenarios.append(("Chuyển sang phân khu kế tiếp (giá mềm hơn)", TIER_TO_LABEL[nb_tier - 1], p_nb, p_nb - pred_usd))

    s_g = user_input.copy(); s_g["GarageCars"] = min(4, garage_cars + 1)
    if garage_cars < 4:
        p_g = model.predict(s_g)[0]
        scenarios.append(("Tăng sức chứa Gara thêm +1 ô tô", f"{garage_cars + 1} xe", p_g, p_g - pred_usd))

    df_scen = pd.DataFrame(scenarios, columns=["Kịch bản điều chỉnh", "Thông số mới", "Giá dự đoán mới ($)", "Chênh lệch ($)"])
    st.dataframe(
        df_scen.style.format({"Giá dự đoán mới ($)": "${:,.0f}", "Chênh lệch ($)": "${:+,.0f}"}),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 2: LỌC & GỢI Ý NHÀ TẠI AMES, IOWA THEO TIÊU CHÍ
# =========================================================
with tab2:
    st.subheader("🔍 Bộ Lọc Tìm Kiếm Nhà Phù Hợp Tại Thành Phố Ames, Iowa")
    st.caption("Lọc danh sách nhà theo khoảng diện tích sq ft, phân khu, số gara ô tô và ngân sách USD của bạn:")

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        min_sf, max_sf = st.slider("Diện tích sàn (sq ft)", 700, 3500, (1300, 2100), step=50)
    with f2:
        max_b_usd = st.number_input("Ngân sách tối đa ($)", 80000, 1500000, int(ngan_sach_usd), step=5000)
    with f3:
        min_qual = st.slider("Chất lượng tối thiểu (1–10)", 1, 10, 6)
    with f4:
        min_gara = st.selectbox("Gara tối thiểu (xe)", [0, 1, 2, 3], index=2)
    with f5:
        min_yr = st.number_input("Xây từ năm trở lại đây", 1920, 2026, 1990, step=5)

    filtered = df_market[
        (df_market["GrLivArea"] >= min_sf) &
        (df_market["GrLivArea"] <= max_sf) &
        (df_market["SalePrice"] <= max_b_usd) &
        (df_market["OverallQual"] >= min_qual) &
        (df_market["GarageCars"] >= min_gara) &
        (df_market["YearBuilt"] >= min_yr)
    ].copy()

    if len(filtered) == 0:
        st.warning("Chưa có căn nhà nào tại Ames khớp đồng thời mọi điều kiện lọc. Hãy thử nới rộng khoảng sq ft, năm xây dựng hoặc ngân sách.")
    else:
        filtered["DienTich_m2"] = np.round(filtered["GrLivArea"] * SQFT_TO_M2, 1)
        filtered["DonGia_SqFt"] = np.round(filtered["SalePrice"] / filtered["GrLivArea"], 1)
        filtered["DuNganSach_USD"] = np.round(max_b_usd - filtered["SalePrice"], 0)

        s_area = (filtered["GrLivArea"] - min_sf) / max(1, (max_sf - min_sf)) * 25
        s_loc = (filtered["NeighborhoodTier"] / 5) * 25
        s_qual = (filtered["OverallQual"] / 10) * 25
        s_bsmt = np.minimum(filtered["TotalBsmtSF"] / 1500, 1.0) * 15
        s_yr = np.clip((filtered["YearBuilt"] - 1950) / 76, 0, 1) * 10
        filtered["DiemPhuHop"] = np.round(s_area + s_loc + s_qual + s_bsmt + s_yr, 1)

        top_matches = filtered.sort_values("DiemPhuHop", ascending=False).head(20).copy()
        top_matches["PhanKhu"] = top_matches["NeighborhoodTier"].map(TIER_TO_LABEL)

        st.success(f"Tìm thấy **{len(filtered)}** căn nhà tại Ames, IA thỏa mãn tiêu chí! Dưới đây là **Top {len(top_matches)} lựa chọn tối ưu nhất**:")

        display_cols = [
            "DiemPhuHop", "PhanKhu", "GrLivArea", "DienTich_m2", "TotalBsmtSF",
            "SalePrice", "DonGia_SqFt", "OverallQual", "YearBuilt", "BedroomAbvGr",
            "GarageCars", "DuNganSach_USD"
        ]
        rename_cols = {
            "DiemPhuHop": "Điểm (/100)",
            "PhanKhu": "Phân khu Ames",
            "GrLivArea": "Sàn ở (sq ft)",
            "DienTich_m2": "Quy đổi (m²)",
            "TotalBsmtSF": "Tầng hầm (sq ft)",
            "SalePrice": "Giá bán ($)",
            "DonGia_SqFt": "$ / sq ft",
            "OverallQual": "Chất lượng",
            "YearBuilt": "Năm XD",
            "BedroomAbvGr": "Phòng ngủ",
            "GarageCars": "Gara (xe)",
            "DuNganSach_USD": "Dư ngân sách ($)"
        }
        st.dataframe(top_matches[display_cols].rename(columns=rename_cols), use_container_width=True, hide_index=True)

# =========================================================
# TAB 3: TÍNH VAY MORTGAGE CHUẨN MỸ (PITI + THUẾ STORY COUNTY, IA)
# =========================================================
with tab3:
    st.subheader("🏦 Mô Phỏng Vay Mua Nhà Tại Mỹ (US Mortgage PITI Calculator)")
    st.caption(
        f"Tính toán chi phí hàng tháng thực tế tại Ames, Story County, Iowa cho căn nhà trị giá **${pred_usd:,.0f}** "
        "(bao gồm Gốc + Lãi + Thuế bất động sản bang Iowa ~1.55%/năm + Bảo hiểm nhà + PMI nếu trả trước < 20%)."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        down_pct = st.slider("Trả trước (Down Payment %)", 5, 80, 20, step=5)
    with m2:
        loan_years = st.selectbox("Kỳ hạn vay cố định (Fixed-Rate)", [30, 15, 20], index=0)
    with m3:
        apr = st.number_input("Lãi suất vay Mortgage (%/năm)", 3.5, 11.0, 6.4, step=0.1)
    with m4:
        gross_monthly_income = st.number_input("Thu nhập hộ gia đình ($/tháng)", 2500, 40000, 9500, step=250)

    down_payment_usd = pred_usd * (down_pct / 100.0)
    loan_amount_usd = pred_usd - down_payment_usd
    n_months = loan_years * 12
    r_month = (apr / 100.0) / 12.0

    if loan_amount_usd > 0 and r_month > 0:
        pi_monthly = loan_amount_usd * (r_month * (1 + r_month) ** n_months) / ((1 + r_month) ** n_months - 1)
    else:
        pi_monthly = loan_amount_usd / max(1, n_months)

    prop_tax_monthly = (pred_usd * 0.0155) / 12.0
    home_ins_monthly = 1450.0 / 12.0
    pmi_monthly = (loan_amount_usd * 0.006) / 12.0 if down_pct < 20 else 0.0
    total_piti_monthly = pi_monthly + prop_tax_monthly + home_ins_monthly + pmi_monthly
    dti_front_end = (total_piti_monthly / gross_monthly_income) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Tiền mặt cần chuẩn bị ban đầu",
            f"${down_payment_usd + closing_cost_usd:,.0f}",
            delta=f"Down ${down_payment_usd:,.0f} + Closing ${closing_cost_usd:,.0f}",
            delta_color="off"
        )
    with k2:
        st.metric(
            "Khoản vay ngân hàng (Loan)",
            f"${loan_amount_usd:,.0f}",
            delta=f"LTV: {100 - down_pct}% giá trị nhà",
            delta_color="off"
        )
    with k3:
        st.metric(
            "Tổng trả hàng tháng (PITI)",
            f"${total_piti_monthly:,.0f} / tháng",
            delta=f"Gốc+Lãi ${pi_monthly:,.0f} | Thuế+BH ${prop_tax_monthly + home_ins_monthly + pmi_monthly:,.0f}",
            delta_color="off"
        )
    with k4:
        st.metric(
            "Tỷ lệ PITI / Thu nhập (28% Rule)",
            f"{dti_front_end:.1f}%",
            delta="Chuẩn an toàn Mỹ (≤ 28%)" if dti_front_end <= 28 else "Vượt ngưỡng 28% thu nhập",
            delta_color="normal" if dti_front_end <= 28 else "inverse"
        )

    if down_pct < 20:
        st.info(f"ℹ️ Vì mức trả trước **{down_pct}%** dưới ngưỡng 20%, khoản vay tại Mỹ phát sinh thêm bảo hiểm **PMI ~${pmi_monthly:,.0f}/tháng** cho đến khi bạn trả được 20% giá trị căn nhà.")

    if dti_front_end <= 28:
        st.success(f"✅ **Đạt chuẩn duyệt vay ngân hàng Mỹ ({dti_front_end:.1f}% ≤ 28%):** Tổng chi phí nhà ở hàng tháng **${total_piti_monthly:,.0f}** (đã gồm thuế đất Story County, IA) hoàn toàn nằm trong vùng an toàn tài chính.")
    else:
        st.warning(f"⚠️ **Vượt quy tắc 28% ({dti_front_end:.1f}%):** Ngân hàng Mỹ thường khuyến nghị PITI ≤ 28% thu nhập gộp. Bạn có thể tăng Down Payment lên **20–25%**, chọn kỳ hạn **30 năm** hoặc giảm mức giá mục tiêu.")

    fig_piti = go.Figure(go.Pie(
        labels=["Gốc & Lãi vay (P&I)", "Thuế BĐS Story County (1.55%)", "Bảo hiểm nhà ở", "Bảo hiểm khoản vay (PMI)"],
        values=[round(pi_monthly, 1), round(prop_tax_monthly, 1), round(home_ins_monthly, 1), round(pmi_monthly, 1)],
        hole=0.45,
        marker_colors=["#2563EB", "#0EA5E9", "#10B981", "#F59E0B"]
    ))
    fig_piti.update_layout(
        title=f"Cơ cấu chi phí nhà ở hàng tháng tại Ames, IA (Tổng: ${total_piti_monthly:,.0f}/tháng)",
        height=350,
        margin=dict(l=20, r=20, t=45, b=20)
    )
    st.plotly_chart(fig_piti, use_container_width=True)

# =========================================================
# TAB 4: CHẤM ĐIỂM & CHECKLIST KIỂM ĐỊNH NHÀ TẠI AMES, IOWA
# =========================================================
with tab4:
    st.subheader("📋 Chấm Điểm Đánh Giá & Checklist Kiểm Định Nhà (Home Inspection) Tại Iowa")
    col_sc1, col_sc2 = st.columns([1.1, 1])

    with col_sc1:
        st.markdown("#### 1. Biểu đồ đánh giá chất lượng 5 trục")
        s_loc = round((nb_tier / 5) * 100, 1)
        s_qual = round((overall_qual / 10) * 100, 1)
        s_bsmt = round(min(100.0, (total_bsmt_sf / max(1, gr_liv_area)) * 115), 1)
        s_winter = round(min(100.0, (garage_cars / 3) * 65 + (35 if central_air else 0)), 1)
        s_fin = 95.0 if budget_diff_usd >= 8000 else (75.0 if budget_diff_usd >= 0 else 45.0)

        categories = ["Vị trí phân khu Ames", "Chất lượng vật liệu", "Tầng hầm (Basement)", "Tiện nghi mùa đông IA", "Phù hợp tài chính"]
        scores = [s_loc, s_qual, s_bsmt, s_winter, s_fin]
        avg_score = round(float(np.mean(scores)), 1)

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#2563EB",
            fillcolor="rgba(37, 99, 235, 0.2)"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
            title=f"Điểm tổng hợp căn nhà: {avg_score} / 100 điểm"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_sc2:
        st.markdown("#### 2. Checklist 8 bước bắt buộc khi mua nhà tại Ames, IA")
        st.caption("Các hạng mục đặc thù vùng Trung Tây Hoa Kỳ cần kiểm tra trước khi chốt hợp đồng:")
        checks = [
            st.checkbox("1. Đo nồng độ khí Radon (Radon Test) — Bắt buộc vì bang Iowa có mức Radon tự nhiên cao"),
            st.checkbox("2. Kiểm tra nền móng tầng hầm (Basement), vết nứt tường & máy bơm chống ngập (Sump Pump)"),
            st.checkbox("3. Kiểm tra lò sưởi (Furnace) & điều hòa HVAC hoạt động tốt cho mùa đông âm độ ở Iowa"),
            st.checkbox("4. Kiểm tra mái ngói (Roof) & máng xối xem có hư hại do mưa đá (Hailstorm)"),
            st.checkbox("5. Tra cứu bản đồ ngập lụt FEMA Flood Map (đặc biệt khu gần sông Skunk & suối Squaw Creek)"),
            st.checkbox("6. Kiểm tra hệ thống ống thoát nước chính (Sewer Scope) với nhà xây trước năm 1980"),
            st.checkbox("7. Kiểm tra quy định HOA (nếu có) và phân tuyến trường học (Ames Community School District)"),
            st.checkbox("8. Kiểm tra lịch sử thuế đất trên cổng thông tin Story County Assessor & bảo hiểm quyền sở hữu (Title Insurance)")
        ]
        done_count = sum(checks)
        st.progress(done_count / len(checks))
        st.write(f"**Tiến độ kiểm định (Home Inspection):** {done_count} / {len(checks)} hạng mục ({int(done_count/len(checks)*100)}%)")
        if done_count == len(checks):
            st.success("🎉 Hoàn tất đầy đủ quy trình kiểm định an toàn chuẩn bất động sản Ames, Iowa!")

# =========================================================
# TAB 5: SO SÁNH CÁC CĂN NHÀ ĐÃ LƯU TẠI AMES
# =========================================================
with tab5:
    st.subheader("⚖️ Bảng So Sánh Các Căn Nhà Đang Cân Nhắc Tại Ames, Iowa")
    st.write("Thay đổi thông số bên trái và nhấn **'➕ Lưu căn nhà này vào bảng so sánh'** ở Tab 1 để đối chiếu nhiều phương án.")

    if len(st.session_state.saved_ames_houses) == 0:
        st.info("Chưa có căn nhà nào được lưu. Hãy quay lại **Tab 1**, chọn thông số và bấm nút lưu.")
    else:
        df_compare = pd.DataFrame(st.session_state.saved_ames_houses)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns([3, 1])
        with c_left:
            x_labels = [f"#{i+1} {name}" for i, name in enumerate(df_compare["Tên phương án"])]
            gia_vals = pd.to_numeric(df_compare["Giá dự đoán ($)"], errors="coerce").fillna(0)
            tron_goi_vals = pd.to_numeric(df_compare["Trọn gói ($)"], errors="coerce").fillna(0)
            du_thieu_vals = pd.to_numeric(df_compare["Dư/Thiếu NS ($)"], errors="coerce").fillna(0)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=x_labels, y=gia_vals, name="Giá dự đoán ($)",
                marker_color="#2563EB",
                text=[f"${v:,.0f}" for v in gia_vals], textposition="outside"
            ))
            fig_cmp.add_trace(go.Bar(
                x=x_labels, y=tron_goi_vals, name="Trọn gói ($)",
                marker_color="#0284C7",
                text=[f"${v:,.0f}" for v in tron_goi_vals], textposition="outside"
            ))
            fig_cmp.add_trace(go.Bar(
                x=x_labels, y=du_thieu_vals, name="Dư/Thiếu NS ($)",
                marker_color=["#16A34A" if v >= 0 else "#DC2626" for v in du_thieu_vals],
                text=[f"${v:+,.0f}" for v in du_thieu_vals], textposition="outside"
            ))
            fig_cmp.update_layout(
                barmode="group",
                title="So sánh Giá bán, Chi phí trọn gói & Mức dư/vượt ngân sách ($ USD)",
                yaxis_title="USD ($)",
                height=390,
                margin=dict(l=20, r=20, t=55, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        with c_right:
            csv_export = df_compare.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải bảng so sánh (CSV)",
                data=csv_export,
                file_name="so_sanh_nha_ames_iowa.csv",
                mime="text/csv",
                use_container_width=True
            )
            if st.button("🗑️ Xóa toàn bộ danh sách", use_container_width=True):
                st.session_state.saved_ames_houses = []
                st.rerun()

# =========================================================
# TAB 6: NẠP DỮ LIỆU AMES HOUSING CỦA NHÓM 5
# =========================================================
with tab6:
    st.subheader("📁 Nạp & Huấn Luyện Trực Tiếp Trên Dữ Liệu Ames Housing Của Nhóm 5")
    st.info(
        "💡 **Tự động nhận diện dữ liệu Ames, Iowa:**\n"
        "- Nếu bạn tải lên tệp **`Nhóm5_GiaNha_2.xlsx`** (tệp tổng 100% dữ liệu) hoặc **`Nhóm5 _GiaNha_train_2.xlsx`** có chứa các cột chuẩn của Ames Housing "
        "(`SalePrice`, `GrLivArea`, `Neighborhood`, `OverallQual`, ...), ứng dụng sẽ **tự động huấn luyện lại mô hình theo dữ liệu thật của bạn**!"
    )

    uploaded = st.file_uploader("Tải lên tệp Excel (.xlsx) hoặc CSV (.csv)", type=["xlsx", "csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.success(f"Đã đọc tệp `{uploaded.name}`: **{len(df_up):,} dòng** × **{len(df_up.columns)} cột**.")
            st.dataframe(df_up.head(15), use_container_width=True)

            if {"SalePrice", "GrLivArea", "OverallQual"}.issubset(df_up.columns):
                clean_df = df_up.copy()
                clean_df["NeighborhoodTier"] = clean_df["Neighborhood"].map(CODE_TO_TIER).fillna(3) if "Neighborhood" in clean_df.columns else 3
                clean_df["LotArea"] = pd.to_numeric(clean_df.get("LotArea", 9000), errors="coerce").fillna(9000)
                clean_df["TotalBsmtSF"] = pd.to_numeric(clean_df.get("TotalBsmtSF", 900), errors="coerce").fillna(900)
                clean_df["YearBuilt"] = pd.to_numeric(clean_df.get("YearBuilt", 2000), errors="coerce").fillna(2000)
                clean_df["BedroomAbvGr"] = pd.to_numeric(clean_df.get("BedroomAbvGr", 3), errors="coerce").fillna(3)
                clean_df["FullBath"] = pd.to_numeric(clean_df.get("FullBath", 2), errors="coerce").fillna(2)
                clean_df["GarageCars"] = pd.to_numeric(clean_df.get("GarageCars", 2), errors="coerce").fillna(2)
                if "CentralAir" in clean_df.columns:
                    clean_df["CentralAir"] = clean_df["CentralAir"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(1)
                else:
                    clean_df["CentralAir"] = 1

                req_cols = [
                    "GrLivArea", "LotArea", "TotalBsmtSF", "NeighborhoodTier",
                    "OverallQual", "YearBuilt", "BedroomAbvGr", "FullBath",
                    "GarageCars", "CentralAir", "SalePrice"
                ]
                clean_df = clean_df[req_cols].dropna()
                if len(clean_df) >= 50:
                    st.session_state.custom_ames_df = clean_df
                    st.success(f"🚀 Đã kích hoạt mô hình huấn luyện trực tiếp trên **{len(clean_df):,} căn nhà** tại Ames, Iowa từ tệp của bạn!")
            elif {"Rank", "Score", "SubmissionCount"}.issubset(df_up.columns):
                st.warning(
                    "⚠️ Tệp này là **bảng xếp hạng cuộc thi Kaggle** (chứa `Rank`, `Score`, `SubmissionCount`), "
                    "chưa có thông số nhà ở Ames, Iowa. App vẫn sử dụng bộ dữ liệu thị trường Ames, IA tích hợp sẵn ở các Tab 1–5."
                )
        except Exception as e:
            st.error(f"Lỗi khi đọc tệp: {e}")
