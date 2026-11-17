import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Trợ Lý Dự Đoán & Chọn Mua Nhà Thông Minh",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MÔ HÌNH RỪNG NGẪU NHIÊN THUẦN NUMPY (KHÔNG CẦN SKLEARN)
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
            thresholds = np.percentile(vals, np.linspace(15, 85, min(12, len(uniq))))
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
# 3. TẠO DỮ LIỆU THỊ TRƯỜNG & HUẤN LUYỆN MÔ HÌNH
# ---------------------------------------------------------
KHU_VUC_MAP = {
    1: "Khu vực 1: Trung tâm VIP / Đô thị lõi",
    2: "Khu vực 2: Nội thành phát triển",
    3: "Khu vực 3: Ven trung tâm / Đang mở rộng",
    4: "Khu vực 4: Ngoại ô / Vùng ven"
}
KHU_VUC_REVERSE = {v: k for k, v in KHU_VUC_MAP.items()}

NOI_THAT_MAP = {
    1: "Cơ bản (Nhà thô / Sửa chữa nhẹ)",
    2: "Khá (Đầy đủ nội thất tiêu chuẩn)",
    3: "Cao cấp (Thiết kế hiện đại, vào ở ngay)"
}
NOI_THAT_REVERSE = {v: k for k, v in NOI_THAT_MAP.items()}


@st.cache_data
def generate_market_data(n_samples: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    area_m2 = np.round(rng.uniform(30, 220, n_samples), 1)
    floors = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.14, 0.26, 0.30, 0.20, 0.10])
    bedrooms = np.clip(np.round(area_m2 / 26 + floors * 0.45 + rng.normal(0, 0.5, n_samples)), 1, 8).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.85 + rng.normal(0, 0.35, n_samples)), 1, 8).astype(int)
    location_tier = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.30, 0.35, 0.20])
    alley_width = rng.choice([2.0, 3.0, 4.5, 6.0, 8.0, 12.0], size=n_samples, p=[0.12, 0.22, 0.24, 0.18, 0.14, 0.10])
    house_age = rng.integers(0, 25, size=n_samples)
    interior_quality = rng.choice([1, 2, 3], size=n_samples, p=[0.28, 0.47, 0.25])
    has_garage = ((alley_width >= 4.5) & (rng.random(n_samples) > 0.35)).astype(int)
    legal_status = rng.choice([1, 0], size=n_samples, p=[0.90, 0.10])
    near_amenities = rng.choice([1, 0], size=n_samples, p=[0.65, 0.35])

    base_unit = np.select(
        [location_tier == 1, location_tier == 2, location_tier == 3, location_tier == 4],
        [0.140, 0.092, 0.058, 0.032]
    )
    road_mult = 1.0 + (alley_width - 3.5) * 0.032
    construction_val = area_m2 * floors * 0.0075
    room_val = bedrooms * 0.10 + bathrooms * 0.07
    quality_mult = np.select(
        [interior_quality == 1, interior_quality == 2, interior_quality == 3],
        [0.92, 1.00, 1.14]
    )
    age_mult = np.maximum(0.76, 1.0 - house_age * 0.0065)
    garage_val = has_garage * 0.35
    amenity_mult = np.where(near_amenities == 1, 1.05, 0.97)
    legal_mult = np.where(legal_status == 1, 1.00, 0.78)
    noise = rng.normal(1.0, 0.03, n_samples)

    price_ty = (
        (area_m2 * base_unit * road_mult + construction_val + room_val + garage_val)
        * quality_mult * age_mult * amenity_mult * legal_mult * noise
    )
    price_ty = np.round(np.maximum(0.8, price_ty), 2)

    return pd.DataFrame({
        "DienTich_m2": area_m2,
        "KhuVuc": location_tier,
        "DoRongDuong_m": alley_width,
        "SoTang": floors,
        "SoPhongNgu": bedrooms,
        "SoPhongWC": bathrooms,
        "NoiThat": interior_quality,
        "TuoiNha_Nam": house_age,
        "CoGaraOto": has_garage,
        "GanTienIch": near_amenities,
        "PhapLy_SoHong": legal_status,
        "GiaNha_TyVND": price_ty
    })


@st.cache_resource
def train_house_model(df: pd.DataFrame):
    feature_cols = [
        "DienTich_m2", "KhuVuc", "DoRongDuong_m", "SoTang",
        "SoPhongNgu", "SoPhongWC", "NoiThat", "TuoiNha_Nam",
        "CoGaraOto", "GanTienIch", "PhapLy_SoHong"
    ]
    X = df[feature_cols].values
    y = df["GiaNha_TyVND"].values

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


if "saved_houses" not in st.session_state:
    st.session_state.saved_houses = []

df_market = generate_market_data()
model, feature_cols, model_r2, model_mae = train_house_model(df_market)

# ---------------------------------------------------------
# 4. THANH BÊN (SIDEBAR): CHỌN LỌC THÔNG SỐ NGÔI NHÀ CỤ THỂ
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎯 Thiết Lập Thông Số Ngôi Nhà")
    st.caption("Nhập chính xác diện tích (m²) và các tiêu chí bạn đang cân nhắc:")

    ten_phuong_an = st.text_input("Tên gợi nhớ phương án", value="Phương án 1 (Ưu tiên)")

    ngan_sach = st.number_input(
        "💰 Ngân sách dự kiến của bạn (Tỷ VNĐ)",
        min_value=0.5, max_value=100.0, value=5.5, step=0.1,
        help="Dùng để đối chiếu xem ngôi nhà có nằm trong khả năng tài chính của bạn không."
    )

    st.divider()
    st.subheader("1. Diện tích & Vị trí")
    dien_tich = st.number_input(
        "📐 Diện tích cụ thể muốn mua (m²)",
        min_value=20.0, max_value=500.0, value=65.0, step=1.0,
        help="Nhập chính xác số mét vuông đất/sàn bạn dự định mua."
    )

    khu_vuc_label = st.selectbox(
        "📍 Khu vực tọa lạc",
        options=list(KHU_VUC_REVERSE.keys()),
        index=1
    )
    khu_vuc = KHU_VUC_REVERSE[khu_vuc_label]

    do_rong_duong = st.slider(
        "🛣️ Độ rộng đường / hẻm trước nhà (m)",
        min_value=1.5, max_value=20.0, value=4.5, step=0.5
    )

    st.subheader("2. Quy mô & Công năng")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        so_tang = st.selectbox("🏢 Số tầng", options=[1, 2, 3, 4, 5], index=2)
        so_phong_ngu = st.number_input("🛏️ Phòng ngủ", min_value=1, max_value=10, value=3, step=1)
    with col_s2:
        so_phong_wc = st.number_input("🚿 Phòng WC", min_value=1, max_value=10, value=3, step=1)
        tuoi_nha = st.number_input("⏳ Tuổi nhà (năm)", min_value=0, max_value=40, value=2, step=1)

    st.subheader("3. Chất lượng & Tiện ích")
    noi_that_label = st.selectbox(
        "🛋️ Mức độ hoàn thiện nội thất",
        options=list(NOI_THAT_REVERSE.keys()),
        index=1
    )
    noi_that = NOI_THAT_REVERSE[noi_that_label]

    co_gara = st.checkbox("🚗 Có gara / chỗ đỗ ô tô trong nhà", value=(do_rong_duong >= 4.5))
    gan_tien_ich = st.checkbox("🏫 Gần trường học, bệnh viện, chợ/siêu thị", value=True)
    phap_ly = st.checkbox("📜 Pháp lý chuẩn (Sổ hồng / Sổ đỏ riêng)", value=True)

user_input = pd.DataFrame([{
    "DienTich_m2": float(dien_tich),
    "KhuVuc": int(khu_vuc),
    "DoRongDuong_m": float(do_rong_duong),
    "SoTang": int(so_tang),
    "SoPhongNgu": int(so_phong_ngu),
    "SoPhongWC": int(so_phong_wc),
    "NoiThat": int(noi_that),
    "TuoiNha_Nam": int(tuoi_nha),
    "CoGaraOto": int(co_gara),
    "GanTienIch": int(gan_tien_ich),
    "PhapLy_SoHong": int(phap_ly)
}])

pred_price, low_price, high_price = predict_with_interval(model, user_input)
unit_price_trieu = (pred_price * 1000) / dien_tich
chenh_lech_ngan_sach = ngan_sach - pred_price

# ---------------------------------------------------------
# 5. GIAO DIỆN CHÍNH & CÁC TAB CHỨC NĂNG
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏠 Hệ Thống Dự Đoán & Tư Vấn Chọn Mua Nhà Đa Yếu Tố</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Phân tích định giá chính xác theo từng mét vuông (m²), đánh giá tác động các yếu tố và gợi ý phương án tối ưu ngân sách.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dự Đoán & Phân Tích Yếu Tố",
    "🔍 2. Lọc & Gợi Ý Nhà Theo m²",
    "⚖️ 3. So Sánh Các Phương Án",
    "📁 4. Dữ Liệu & Tải File Riêng"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"Giá dự đoán ({dien_tich:g} m²)",
            value=f"{pred_price:.2f} Tỷ",
            delta=f"Khoảng: {low_price:.2f} - {high_price:.2f} Tỷ",
            delta_color="off"
        )
    with c2:
        st.metric(
            label="Đơn giá trung bình / m²",
            value=f"{unit_price_trieu:.1f} Tr/m²",
            delta=f"{so_tang} tầng | {so_phong_ngu} PN"
        )
    with c3:
        st.metric(
            label="So với ngân sách của bạn",
            value=f"{abs(chenh_lech_ngan_sach):.2f} Tỷ",
            delta="Dư ngân sách (An toàn)" if chenh_lech_ngan_sach >= 0 else "Vượt ngân sách dự kiến",
            delta_color="normal" if chenh_lech_ngan_sach >= 0 else "inverse"
        )
    with c4:
        st.metric(
            label="Độ tin cậy mô hình (R²)",
            value=f"{model_r2 * 100:.1f}%",
            delta=f"Sai số TB: ±{model_mae:.2f} Tỷ",
            delta_color="off"
        )

    if chenh_lech_ngan_sach >= 0.3:
        st.success(
            f"✅ **Đánh giá khả thi:** Với ngân sách **{ngan_sach:.2f} Tỷ**, cấu hình nhà **{dien_tich:g} m²** tại *{khu_vuc_label}* hoàn toàn nằm trong tầm tay (dư khoảng **{chenh_lech_ngan_sach:.2f} Tỷ** để dự phòng chi phí sang tên, sửa chữa hoặc nội thất)."
        )
    elif chenh_lech_ngan_sach >= 0:
        st.info(
            f"ℹ️ **Đánh giá sát ngân sách:** Giá dự đoán (**{pred_price:.2f} Tỷ**) vừa khít ngân sách (**{ngan_sach:.2f} Tỷ**). Bạn nên thương lượng giảm thêm 3–5% hoặc giữ quỹ dự phòng thuế phí."
        )
    else:
        st.warning(
            f"⚠️ **Cảnh báo vượt ngân sách {abs(chenh_lech_ngan_sach):.2f} Tỷ:** Để đạt được đúng diện tích **{dien_tich:g} m²**, bạn có thể cân nhắc chuyển sang khu vực lân cận, giảm số tầng/mức nội thất hoặc xem biểu đồ bên dưới để biết diện tích tối đa phù hợp ngân sách."
        )

    col_btn1, _ = st.columns([1, 3])
    with col_btn1:
        if st.button("➕ Lưu phương án này để so sánh", use_container_width=True):
            st.session_state.saved_houses.append({
                "Tên phương án": ten_phuong_an,
                "Diện tích (m²)": dien_tich,
                "Khu vực": khu_vuc_label.split(":")[0],
                "Đường (m)": do_rong_duong,
                "Số tầng": so_tang,
                "Phòng ngủ": so_phong_ngu,
                "Phòng WC": so_phong_wc,
                "Nội thất": noi_that_label.split(" ")[0],
                "Tuổi nhà": tuoi_nha,
                "Gara": "Có" if co_gara else "Không",
                "Sổ hồng": "Có" if phap_ly else "Chưa",
                "Giá dự đoán (Tỷ)": round(pred_price, 2),
                "Đơn giá (Tr/m²)": round(unit_price_trieu, 1),
                "Chênh lệch NS (Tỷ)": round(chenh_lech_ngan_sach, 2)
            })
            st.toast(f"Đã lưu '{ten_phuong_an}' vào Tab So sánh!", icon="✅")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("📈 Biến thiên giá theo Diện tích (m²)")
        st.caption("Giữ nguyên các tiêu chí khác, thay đổi diện tích từ 30 m² đến 200 m²:")

        area_grid = np.arange(30, 205, 5)
        sim_df = pd.concat([user_input] * len(area_grid), ignore_index=True)
        sim_df["DienTich_m2"] = area_grid
        sim_prices = model.predict(sim_df)

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=area_grid, y=sim_prices,
            mode="lines", name="Đường giá dự đoán",
            line=dict(color="#2563EB", width=3)
        ))
        fig_area.add_trace(go.Scatter(
            x=[dien_tich], y=[pred_price],
            mode="markers+text", name="Lựa chọn hiện tại",
            text=[f"{dien_tich:g} m² ({pred_price:.2f} Tỷ)"],
            textposition="top left",
            marker=dict(color="#DC2626", size=12, symbol="diamond")
        ))
        fig_area.add_hline(
            y=ngan_sach, line_dash="dash", line_color="#16A34A",
            annotation_text=f"Ngân sách: {ngan_sach:.2f} Tỷ",
            annotation_position="bottom right"
        )
        fig_area.update_layout(
            xaxis_title="Diện tích (m²)",
            yaxis_title="Giá dự đoán (Tỷ VNĐ)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        affordable_areas = area_grid[sim_prices <= ngan_sach]
        if len(affordable_areas) > 0:
            st.caption(f"💡 Với cấu hình hiện tại và ngân sách **{ngan_sach:.2f} Tỷ**, bạn có thể mua diện tích lên tới khoảng **{affordable_areas.max()} m²**.")
        else:
            st.caption(f"💡 Với cấu hình hiện tại, ngay cả mức 30 m² cũng vượt ngân sách **{ngan_sach:.2f} Tỷ**.")

    with chart_col2:
        st.subheader("🧩 Mức độ ảnh hưởng của các yếu tố")
        st.caption("Tỷ trọng quyết định giá trị ngôi nhà từ mô hình dự đoán:")

        label_vn = {
            "DienTich_m2": "Diện tích (m²)",
            "KhuVuc": "Vị trí / Khu vực",
            "DoRongDuong_m": "Độ rộng đường/hẻm",
            "SoTang": "Số tầng xây dựng",
            "SoPhongNgu": "Số phòng ngủ",
            "SoPhongWC": "Số phòng WC",
            "NoiThat": "Chất lượng nội thất",
            "TuoiNha_Nam": "Tuổi thọ ngôi nhà",
            "CoGaraOto": "Gara ô tô",
            "GanTienIch": "Gần tiện ích công cộng",
            "PhapLy_SoHong": "Pháp lý (Sổ hồng)"
        }
        imp_df = pd.DataFrame({
            "Yếu tố": [label_vn[c] for c in feature_cols],
            "Tầm quan trọng (%)": model.feature_importances_ * 100
        }).sort_values("Tầm quan trọng (%)", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Tầm quan trọng (%)", y="Yếu tố", orientation="h",
            text=imp_df["Tầm quan trọng (%)"].apply(lambda x: f"{x:.1f}%"),
            color="Tầm quan trọng (%)", color_continuous_scale="Blues"
        )
        fig_imp.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False,
            xaxis_title="Mức độ tác động (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("🔄 Phân Tích Độ Nhạy: Nếu Thay Đổi Từng Yếu Tố Thì Giá Biến Động Ra Sao?")
    scenarios = []
    s_plus10 = user_input.copy(); s_plus10["DienTich_m2"] += 10
    p_plus10 = model.predict(s_plus10)[0]
    scenarios.append(("Tăng thêm +10 m² diện tích", f"{dien_tich + 10:g} m²", p_plus10, p_plus10 - pred_price))

    if dien_tich > 30:
        s_minus10 = user_input.copy(); s_minus10["DienTich_m2"] = max(20.0, dien_tich - 10)
        p_minus10 = model.predict(s_minus10)[0]
        scenarios.append(("Giảm bớt -10 m² diện tích", f"{s_minus10['DienTich_m2'].iloc[0]:g} m²", p_minus10, p_minus10 - pred_price))

    if khu_vuc < 4:
        s_kv = user_input.copy(); s_kv["KhuVuc"] = khu_vuc + 1
        p_kv = model.predict(s_kv)[0]
        scenarios.append(("Chuyển sang khu vực kế tiếp (xa trung tâm hơn 1 bậc)", KHU_VUC_MAP[khu_vuc + 1].split(":")[0], p_kv, p_kv - pred_price))

    if so_tang > 1:
        s_tang = user_input.copy(); s_tang["SoTang"] = so_tang - 1
        p_tang = model.predict(s_tang)[0]
        scenarios.append(("Giảm bớt 1 tầng xây dựng", f"{so_tang - 1} tầng", p_tang, p_tang - pred_price))

    if noi_that < 3:
        s_nt = user_input.copy(); s_nt["NoiThat"] = 3
        p_nt = model.predict(s_nt)[0]
        scenarios.append(("Nâng cấp lên nội thất cao cấp", "Cao cấp", p_nt, p_nt - pred_price))

    df_scen = pd.DataFrame(scenarios, columns=["Kịch bản điều chỉnh", "Thông số mới", "Giá dự đoán mới (Tỷ)", "Chênh lệch (Tỷ)"])
    st.dataframe(
        df_scen.style.format({"Giá dự đoán mới (Tỷ)": "{:.2f}", "Chênh lệch (Tỷ)": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("🔍 Bộ Lọc Tìm Kiếm & Gợi Ý Ngôi Nhà Tối Ưu")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        min_m2, max_m2 = st.slider("Khoảng diện tích (m²)", 25, 220, (55, 85))
    with f_col2:
        max_budget = st.number_input("Ngân sách tối đa (Tỷ VNĐ)", 1.0, 100.0, float(ngan_sach), step=0.2)
    with f_col3:
        min_pn = st.selectbox("Số phòng ngủ tối thiểu", [1, 2, 3, 4, 5], index=1)
    with f_col4:
        chi_so_hong = st.selectbox("Yêu cầu pháp lý", ["Chỉ lấy nhà có Sổ hồng", "Tất cả"], index=0)

    filtered = df_market[
        (df_market["DienTich_m2"] >= min_m2) &
        (df_market["DienTich_m2"] <= max_m2) &
        (df_market["GiaNha_TyVND"] <= max_budget) &
        (df_market["SoPhongNgu"] >= min_pn)
    ].copy()

    if chi_so_hong == "Chỉ lấy nhà có Sổ hồng":
        filtered = filtered[filtered["PhapLy_SoHong"] == 1]

    if len(filtered) == 0:
        st.warning("Không tìm thấy căn nhà nào khớp 100% bộ lọc hiện tại. Hãy thử nới rộng ngân sách hoặc điều chỉnh khoảng diện tích (m²).")
    else:
        filtered["DonGia_Tr_m2"] = np.round((filtered["GiaNha_TyVND"] * 1000) / filtered["DienTich_m2"], 1)
        filtered["TietKiem_Ty"] = np.round(max_budget - filtered["GiaNha_TyVND"], 2)

        score_area = (filtered["DienTich_m2"] - min_m2) / max(1, (max_m2 - min_m2)) * 30
        score_loc = (5 - filtered["KhuVuc"]) / 4 * 30
        score_road = np.minimum(filtered["DoRongDuong_m"] / 10, 1.0) * 20
        score_age = np.maximum(0, (25 - filtered["TuoiNha_Nam"]) / 25) * 20
        filtered["DiemPhuHop"] = np.round(score_area + score_loc + score_road + score_age, 1)

        top_matches = filtered.sort_values("DiemPhuHop", ascending=False).head(15).copy()
        top_matches["KhuVuc_Ten"] = top_matches["KhuVuc"].map(lambda x: KHU_VUC_MAP[x].split(":")[0])
        top_matches["NoiThat_Ten"] = top_matches["NoiThat"].map(lambda x: NOI_THAT_MAP[x].split(" ")[0])
        top_matches["Gara"] = top_matches["CoGaraOto"].map({1: "Có", 0: "Không"})

        st.success(f"Tìm thấy **{len(filtered)}** căn nhà phù hợp! Dưới đây là **Top {len(top_matches)} lựa chọn tối ưu nhất**:")

        display_cols = [
            "DiemPhuHop", "DienTich_m2", "GiaNha_TyVND", "DonGia_Tr_m2",
            "KhuVuc_Ten", "DoRongDuong_m", "SoTang", "SoPhongNgu",
            "NoiThat_Ten", "TuoiNha_Nam", "Gara", "TietKiem_Ty"
        ]
        rename_cols = {
            "DiemPhuHop": "Điểm phù hợp (/100)",
            "DienTich_m2": "Diện tích (m²)",
            "GiaNha_TyVND": "Giá bán (Tỷ)",
            "DonGia_Tr_m2": "Đơn giá (Tr/m²)",
            "KhuVuc_Ten": "Khu vực",
            "DoRongDuong_m": "Đường (m)",
            "SoTang": "Tầng",
            "SoPhongNgu": "Phòng ngủ",
            "NoiThat_Ten": "Nội thất",
            "TuoiNha_Nam": "Tuổi nhà",
            "Gara": "Gara ô tô",
            "TietKiem_Ty": "Dư ngân sách (Tỷ)"
        }
        st.dataframe(top_matches[display_cols].rename(columns=rename_cols), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("⚖️ Bảng So Sánh Trực Tiếp Các Ngôi Nhà Đang Cân Nhắc")
    if len(st.session_state.saved_houses) == 0:
        st.info("Chưa có phương án nào được lưu. Hãy quay lại **Tab 1**, chọn thông số cho từng căn nhà và bấm nút lưu.")
    else:
        df_compare = pd.DataFrame(st.session_state.saved_houses)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns([3, 1])
        with c_left:
            fig_cmp = px.bar(
                df_compare, x="Tên phương án", y=["Giá dự đoán (Tỷ)", "Chênh lệch NS (Tỷ)"],
                barmode="group", title="So sánh Giá dự đoán & Mức dư/vượt ngân sách"
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        with c_right:
            csv_export = df_compare.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải bảng so sánh (CSV)",
                data=csv_export,
                file_name="so_sanh_phuong_an_mua_nha.csv",
                mime="text/csv",
                use_container_width=True
            )
            if st.button("🗑️ Xóa toàn bộ danh sách", use_container_width=True):
                st.session_state.saved_houses = []
                st.rerun()

with tab4:
    st.subheader("📁 Dữ Liệu Huấn Luyện & Tùy Chọn Tải Dữ Liệu Thực Tế")
    uploaded_file = st.file_uploader("Tải lên tệp CSV dữ liệu nhà đất", type=["csv"])
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"Đã đọc thành công tệp `{uploaded_file.name}` ({len(user_df)} dòng, {len(user_df.columns)} cột).")
            st.dataframe(user_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Không thể đọc tệp: {e}")

    st.markdown("#### Tổng quan dữ liệu thị trường mặc định")
    sample_df = df_market.sample(400, random_state=42)
    fig_scatter = px.scatter(
        sample_df,
        x="DienTich_m2", y="GiaNha_TyVND",
        color=sample_df["KhuVuc"].map(lambda x: KHU_VUC_MAP[x].split(":")[0]),
        size="SoTang",
        labels={
            "DienTich_m2": "Diện tích (m²)",
            "GiaNha_TyVND": "Giá nhà (Tỷ VNĐ)",
            "color": "Khu vực",
            "SoTang": "Số tầng"
        },
        title="Tương quan Diện tích (m²) - Giá nhà theo Khu vực & Số tầng"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
