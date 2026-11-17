import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG & GIAO DIỆN MƯỢT MÀ
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

# ---------------------------------------------------------
# 2. MÔ HÌNH RỪNG NGẪU NHIÊN THUẦN NUMPY (TỐC ĐỘ CAO, 0 LỖI)
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
# 3. DANH MỤC THÔNG SỐ & DỮ LIỆU THỊ TRƯỜNG
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

HUONG_NHA_MAP = {
    3: "Đông Nam / Nam (Mát mẻ, đón gió tốt)",
    2: "Đông / Bắc / Đông Bắc / Tây Nam (Ôn hòa)",
    1: "Tây / Tây Bắc (Nắng chiều, cần chống nóng)"
}
HUONG_NHA_REVERSE = {v: k for k, v in HUONG_NHA_MAP.items()}


@st.cache_data
def generate_market_data(n_samples: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    area_m2 = np.round(rng.uniform(30, 220, n_samples), 1)
    frontage_m = np.round(np.clip(np.sqrt(area_m2) * rng.uniform(0.45, 0.85, n_samples), 3.0, 15.0), 1)
    floors = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.14, 0.26, 0.30, 0.20, 0.10])
    bedrooms = np.clip(np.round(area_m2 / 26 + floors * 0.45 + rng.normal(0, 0.5, n_samples)), 1, 8).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.85 + rng.normal(0, 0.35, n_samples)), 1, 8).astype(int)
    location_tier = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.30, 0.35, 0.20])
    alley_width = rng.choice([2.0, 3.0, 4.5, 6.0, 8.0, 12.0], size=n_samples, p=[0.12, 0.22, 0.24, 0.18, 0.14, 0.10])
    house_age = rng.integers(0, 25, size=n_samples)
    interior_quality = rng.choice([1, 2, 3], size=n_samples, p=[0.28, 0.47, 0.25])
    orientation_score = rng.choice([3, 2, 1], size=n_samples, p=[0.40, 0.40, 0.20])
    has_garage = ((alley_width >= 4.5) & (frontage_m >= 4.0) & (rng.random(n_samples) > 0.30)).astype(int)
    legal_status = rng.choice([1, 0], size=n_samples, p=[0.90, 0.10])
    near_amenities = rng.choice([1, 0], size=n_samples, p=[0.65, 0.35])

    base_unit = np.select(
        [location_tier == 1, location_tier == 2, location_tier == 3, location_tier == 4],
        [0.140, 0.092, 0.058, 0.032]
    )
    road_mult = 1.0 + (alley_width - 3.5) * 0.030
    frontage_mult = 1.0 + (frontage_m - 4.2) * 0.022
    construction_val = area_m2 * floors * 0.0075
    room_val = bedrooms * 0.10 + bathrooms * 0.07
    quality_mult = np.select([interior_quality == 1, interior_quality == 2, interior_quality == 3], [0.92, 1.00, 1.14])
    orient_mult = np.select([orientation_score == 3, orientation_score == 2, orientation_score == 1], [1.03, 1.00, 0.96])
    age_mult = np.maximum(0.76, 1.0 - house_age * 0.0065)
    garage_val = has_garage * 0.35
    amenity_mult = np.where(near_amenities == 1, 1.05, 0.97)
    legal_mult = np.where(legal_status == 1, 1.00, 0.78)
    noise = rng.normal(1.0, 0.025, n_samples)

    price_ty = (
        (area_m2 * base_unit * road_mult * frontage_mult + construction_val + room_val + garage_val)
        * quality_mult * orient_mult * age_mult * amenity_mult * legal_mult * noise
    )
    price_ty = np.round(np.maximum(0.8, price_ty), 2)

    return pd.DataFrame({
        "DienTich_m2": area_m2,
        "MatTien_m": frontage_m,
        "KhuVuc": location_tier,
        "DoRongDuong_m": alley_width,
        "SoTang": floors,
        "SoPhongNgu": bedrooms,
        "SoPhongWC": bathrooms,
        "NoiThat": interior_quality,
        "HuongNha": orientation_score,
        "TuoiNha_Nam": house_age,
        "CoGaraOto": has_garage,
        "GanTienIch": near_amenities,
        "PhapLy_SoHong": legal_status,
        "GiaNha_TyVND": price_ty
    })


@st.cache_resource
def train_house_model(df: pd.DataFrame):
    feature_cols = [
        "DienTich_m2", "MatTien_m", "KhuVuc", "DoRongDuong_m", "SoTang",
        "SoPhongNgu", "SoPhongWC", "NoiThat", "HuongNha", "TuoiNha_Nam",
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
    st.caption("Điều chỉnh chính xác diện tích (m²) và các tiêu chí bạn mong muốn:")

    ten_phuong_an = st.text_input("🏷️ Tên gợi nhớ phương án", value="Căn 1 - Ưu tiên")

    ngan_sach = st.number_input(
        "💰 Ngân sách tối đa của bạn (Tỷ VNĐ)",
        min_value=0.5, max_value=100.0, value=5.8, step=0.1
    )

    st.divider()
    st.subheader("1. Kích thước & Vị trí")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        dien_tich = st.number_input(
            "📐 Diện tích đất (m²)",
            min_value=20.0, max_value=500.0, value=65.0, step=1.0
        )
    with col_d2:
        mat_tien = st.number_input(
            "📏 Mặt tiền (m)",
            min_value=2.5, max_value=30.0, value=4.5, step=0.2
        )

    khu_vuc_label = st.selectbox(
        "📍 Khu vực tọa lạc",
        options=list(KHU_VUC_REVERSE.keys()),
        index=1
    )
    khu_vuc = KHU_VUC_REVERSE[khu_vuc_label]

    do_rong_duong = st.slider(
        "🛣️ Độ rộng đường / hẻm trước nhà (m)",
        min_value=1.5, max_value=20.0, value=5.0, step=0.5
    )

    huong_nha_label = st.selectbox(
        "🧭 Hướng nhà chính",
        options=list(HUONG_NHA_REVERSE.keys()),
        index=0
    )
    huong_nha = HUONG_NHA_REVERSE[huong_nha_label]

    st.subheader("2. Quy mô & Công năng")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        so_tang = st.selectbox("🏢 Số tầng", options=[1, 2, 3, 4, 5], index=2)
        so_phong_ngu = st.number_input("🛏️ Phòng ngủ", min_value=1, max_value=10, value=3, step=1)
    with col_s2:
        so_phong_wc = st.number_input("🚿 Phòng WC", min_value=1, max_value=10, value=3, step=1)
        tuoi_nha = st.number_input("⏳ Tuổi nhà (năm)", min_value=0, max_value=40, value=2, step=1)

    st.subheader("3. Nội thất & Pháp lý")
    noi_that_label = st.selectbox(
        "🛋️ Mức độ hoàn thiện nội thất",
        options=list(NOI_THAT_REVERSE.keys()),
        index=1
    )
    noi_that = NOI_THAT_REVERSE[noi_that_label]

    co_gara = st.checkbox("🚗 Có gara / chỗ đỗ ô tô trong nhà", value=(do_rong_duong >= 4.5 and mat_tien >= 4.0))
    gan_tien_ich = st.checkbox("🏫 Gần trường học, bệnh viện, chợ/siêu thị", value=True)
    phap_ly = st.checkbox("📜 Pháp lý chuẩn (Sổ hồng / Sổ đỏ riêng)", value=True)

user_input = pd.DataFrame([{
    "DienTich_m2": float(dien_tich),
    "MatTien_m": float(mat_tien),
    "KhuVuc": int(khu_vuc),
    "DoRongDuong_m": float(do_rong_duong),
    "SoTang": int(so_tang),
    "SoPhongNgu": int(so_phong_ngu),
    "SoPhongWC": int(so_phong_wc),
    "NoiThat": int(noi_that),
    "HuongNha": int(huong_nha),
    "TuoiNha_Nam": int(tuoi_nha),
    "CoGaraOto": int(co_gara),
    "GanTienIch": int(gan_tien_ich),
    "PhapLy_SoHong": int(phap_ly)
}])

pred_price, low_price, high_price = predict_with_interval(model, user_input)
tong_dien_tich_san = dien_tich * so_tang
unit_price_dat = (pred_price * 1000) / dien_tich
unit_price_san = (pred_price * 1000) / tong_dien_tich_san
phi_sang_ten_uoc_tinh = round(pred_price * 0.025, 2)
tong_chi_phi_lan_banh = round(pred_price + phi_sang_ten_uoc_tinh, 2)
chenh_lech_ngan_sach = ngan_sach - tong_chi_phi_lan_banh

# ---------------------------------------------------------
# 5. GIAO DIỆN CHÍNH & 6 TAB ỨNG DỤNG TOÀN DIỆN
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏠 Trợ Lý Định Giá & Tư Vấn Chọn Mua Nhà Thông Minh</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Định giá chính xác theo từng m², tính toán phương án vay trả góp, chấm điểm khảo sát thực tế và so sánh đa phương án.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Định Giá & Phân Tích m²",
    "🔍 2. Lọc & Gợi Ý Nhà Phù Hợp",
    "🏦 3. Tính Vay & Trả Góp",
    "📋 4. Chấm Điểm & Checklist",
    "⚖️ 5. So Sánh Phương Án",
    "📁 6. Dữ Liệu & Tải File"
])

# =========================================================
# TAB 1: ĐỊNH GIÁ & PHÂN TÍCH ĐA YẾU TỐ
# =========================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"Giá nhà dự đoán ({dien_tich:g} m² đất)",
            value=f"{pred_price:.2f} Tỷ",
            delta=f"Khoảng: {low_price:.2f} - {high_price:.2f} Tỷ",
            delta_color="off"
        )
    with c2:
        st.metric(
            label="Tổng chi phí trọn gói (+2.5% phí)",
            value=f"{tong_chi_phi_lan_banh:.2f} Tỷ",
            delta=f"Thuế/phí sang tên: ~{phi_sang_ten_uoc_tinh:.2f} Tỷ",
            delta_color="off"
        )
    with c3:
        st.metric(
            label=f"Đơn giá ({tong_dien_tich_san:g} m² sàn)",
            value=f"{unit_price_dat:.1f} Tr/m² đất",
            delta=f"~{unit_price_san:.1f} Tr/m² sàn sử dụng",
            delta_color="off"
        )
    with c4:
        st.metric(
            label="Đối chiếu ngân sách trọn gói",
            value=f"{abs(chenh_lech_ngan_sach):.2f} Tỷ",
            delta="Dư ngân sách (An toàn)" if chenh_lech_ngan_sach >= 0 else "Vượt ngân sách dự kiến",
            delta_color="normal" if chenh_lech_ngan_sach >= 0 else "inverse"
        )

    if chenh_lech_ngan_sach >= 0.3:
        st.success(
            f"✅ **Khả thi cao:** Với ngân sách **{ngan_sach:.2f} Tỷ**, căn nhà **{dien_tich:g} m²** (mặt tiền **{mat_tien:g} m**, {so_tang} tầng, tổng **{tong_dien_tich_san:g} m² sàn**) tại *{khu_vuc_label}* hoàn toàn nằm trong khả năng tài chính, còn dư khoảng **{chenh_lech_ngan_sach:.2f} Tỷ** để dự phòng sắm sửa thêm."
        )
    elif chenh_lech_ngan_sach >= 0:
        st.info(
            f"ℹ️ **Sát ngân sách:** Tổng chi phí trọn gói (**{tong_chi_phi_lan_banh:.2f} Tỷ**) vừa khít ngân sách (**{ngan_sach:.2f} Tỷ**). Bạn nên đàm phán bớt 3–5% giá bán hoặc chuyển sang **Tab 3** để tính phương án vay bổ sung."
        )
    else:
        st.warning(
            f"⚠️ **Vượt ngân sách {abs(chenh_lech_ngan_sach):.2f} Tỷ:** Để mua đúng diện tích **{dien_tich:g} m²**, bạn có thể cân nhắc giảm 1 tầng xây dựng, chọn khu vực kế tiếp hoặc xem biểu đồ bên dưới để biết diện tích tối đa mua được trong tầm tiền."
        )

    col_btn1, _ = st.columns([1.4, 3])
    with col_btn1:
        if st.button("➕ Lưu phương án này vào bảng so sánh", use_container_width=True):
            st.session_state.saved_houses.append({
                "Tên phương án": ten_phuong_an,
                "DT đất (m²)": dien_tich,
                "DT sàn (m²)": tong_dien_tich_san,
                "Mặt tiền (m)": mat_tien,
                "Khu vực": khu_vuc_label.split(":")[0],
                "Đường (m)": do_rong_duong,
                "Hướng": huong_nha_label.split("(")[0].strip(),
                "Tầng / PN": f"{so_tang}T - {so_phong_ngu}PN",
                "Sổ hồng": "Có" if phap_ly else "Chưa",
                "Giá dự đoán (Tỷ)": round(pred_price, 2),
                "Trọn gói (Tỷ)": tong_chi_phi_lan_banh,
                "Đơn giá (Tr/m²)": round(unit_price_dat, 1),
                "Dư/Thiếu NS (Tỷ)": round(chenh_lech_ngan_sach, 2)
            })
            st.toast(f"Đã lưu '{ten_phuong_an}' sang Tab 5!", icon="✅")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("📈 Biến thiên giá theo Diện tích (m²)")
        st.caption("Giữ nguyên các tiêu chí khác, khảo sát mức giá từ 30 m² đến 200 m²:")

        area_grid = np.arange(30, 205, 5)
        sim_df = pd.concat([user_input] * len(area_grid), ignore_index=True)
        sim_df["DienTich_m2"] = area_grid
        sim_prices = model.predict(sim_df)

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=area_grid, y=sim_prices,
            mode="lines", name="Giá nhà dự đoán",
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
            xaxis_title="Diện tích đất (m²)",
            yaxis_title="Giá dự đoán (Tỷ VNĐ)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_area, use_container_width=True)

        affordable_areas = area_grid[sim_prices <= ngan_sach]
        if len(affordable_areas) > 0:
            st.caption(f"💡 Với cấu hình hiện tại và ngân sách **{ngan_sach:.2f} Tỷ**, diện tích tối đa bạn có thể chọn là khoảng **{affordable_areas.max()} m²**.")
        else:
            st.caption(f"💡 Với cấu hình hiện tại, mức 30 m² cũng vượt ngân sách **{ngan_sach:.2f} Tỷ**.")

    with chart_col2:
        st.subheader("🧩 Mức độ ảnh hưởng của 13 yếu tố")
        st.caption(f"Độ chính xác mô hình trên dữ liệu kiểm thử: R² = {model_r2*100:.1f}% (Sai số TB ±{model_mae:.2f} Tỷ):")

        label_vn = {
            "DienTich_m2": "Diện tích đất (m²)",
            "MatTien_m": "Bề rộng mặt tiền (m)",
            "KhuVuc": "Vị trí / Khu vực",
            "DoRongDuong_m": "Độ rộng đường/hẻm",
            "SoTang": "Số tầng xây dựng",
            "SoPhongNgu": "Số phòng ngủ",
            "SoPhongWC": "Số phòng WC",
            "NoiThat": "Chất lượng nội thất",
            "HuongNha": "Hướng nhà",
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
            xaxis_title="Tỷ trọng tác động (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("🔄 Bảng Giả Định: Nếu Điều Chỉnh 1 Tiêu Chí Thì Giá Thay Đổi Bao Nhiêu?")
    scenarios = []
    s_plus10 = user_input.copy(); s_plus10["DienTich_m2"] += 10
    p_plus10 = model.predict(s_plus10)[0]
    scenarios.append(("Tăng thêm +10 m² diện tích", f"{dien_tich + 10:g} m²", p_plus10, p_plus10 - pred_price))

    if dien_tich > 30:
        s_minus10 = user_input.copy(); s_minus10["DienTich_m2"] = max(20.0, dien_tich - 10)
        p_minus10 = model.predict(s_minus10)[0]
        scenarios.append(("Giảm bớt -10 m² diện tích", f"{s_minus10['DienTich_m2'].iloc[0]:g} m²", p_minus10, p_minus10 - pred_price))

    s_mt = user_input.copy(); s_mt["MatTien_m"] = mat_tien + 1.0
    p_mt = model.predict(s_mt)[0]
    scenarios.append(("Mở rộng mặt tiền thêm +1.0 m", f"{mat_tien + 1.0:g} m", p_mt, p_mt - pred_price))

    if khu_vuc < 4:
        s_kv = user_input.copy(); s_kv["KhuVuc"] = khu_vuc + 1
        p_kv = model.predict(s_kv)[0]
        scenarios.append(("Chuyển sang khu vực kế tiếp (xa trung tâm hơn)", KHU_VUC_MAP[khu_vuc + 1].split(":")[0], p_kv, p_kv - pred_price))

    if so_tang > 1:
        s_tang = user_input.copy(); s_tang["SoTang"] = so_tang - 1
        p_tang = model.predict(s_tang)[0]
        scenarios.append(("Giảm bớt 1 tầng xây dựng", f"{so_tang - 1} tầng", p_tang, p_tang - pred_price))

    df_scen = pd.DataFrame(scenarios, columns=["Kịch bản điều chỉnh", "Thông số mới", "Giá dự đoán mới (Tỷ)", "Chênh lệch (Tỷ)"])
    st.dataframe(
        df_scen.style.format({"Giá dự đoán mới (Tỷ)": "{:.2f}", "Chênh lệch (Tỷ)": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 2: LỌC & GỢI Ý NHÀ THEO m² & TIÊU CHÍ
# =========================================================
with tab2:
    st.subheader("🔍 Bộ Lọc Tìm Kiếm & Chấm Điểm Nhà Phù Hợp")
    st.caption("Lọc danh sách nhà trên thị trường theo đúng khoảng mét vuông, mặt tiền, hướng nhà và ngân sách của bạn:")

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        min_m2, max_m2 = st.slider("Khoảng diện tích (m²)", 25, 220, (55, 85))
    with f2:
        max_budget = st.number_input("Ngân sách tối đa (Tỷ)", 1.0, 100.0, float(ngan_sach), step=0.2)
    with f3:
        min_mt = st.number_input("Mặt tiền tối thiểu (m)", 3.0, 15.0, 4.0, step=0.5)
    with f4:
        min_pn = st.selectbox("Phòng ngủ tối thiểu", [1, 2, 3, 4, 5], index=1)
    with f5:
        chi_so_hong = st.selectbox("Pháp lý", ["Chỉ lấy Sổ hồng", "Tất cả"], index=0)

    filtered = df_market[
        (df_market["DienTich_m2"] >= min_m2) &
        (df_market["DienTich_m2"] <= max_m2) &
        (df_market["GiaNha_TyVND"] <= max_budget) &
        (df_market["MatTien_m"] >= min_mt) &
        (df_market["SoPhongNgu"] >= min_pn)
    ].copy()

    if chi_so_hong == "Chỉ lấy Sổ hồng":
        filtered = filtered[filtered["PhapLy_SoHong"] == 1]

    if len(filtered) == 0:
        st.warning("Chưa có căn nhà nào khớp đồng thời mọi điều kiện lọc. Hãy thử nới rộng khoảng diện tích (m²) hoặc ngân sách.")
    else:
        filtered["DonGia_Tr_m2"] = np.round((filtered["GiaNha_TyVND"] * 1000) / filtered["DienTich_m2"], 1)
        filtered["TietKiem_Ty"] = np.round(max_budget - filtered["GiaNha_TyVND"], 2)

        score_area = (filtered["DienTich_m2"] - min_m2) / max(1, (max_m2 - min_m2)) * 25
        score_loc = (5 - filtered["KhuVuc"]) / 4 * 25
        score_road = np.minimum(filtered["DoRongDuong_m"] / 10, 1.0) * 20
        score_front = np.minimum(filtered["MatTien_m"] / 8, 1.0) * 15
        score_orient = (filtered["HuongNha"] / 3) * 15
        filtered["DiemPhuHop"] = np.round(score_area + score_loc + score_road + score_front + score_orient, 1)

        top_matches = filtered.sort_values("DiemPhuHop", ascending=False).head(20).copy()
        top_matches["KhuVuc_Ten"] = top_matches["KhuVuc"].map(lambda x: KHU_VUC_MAP[x].split(":")[0])
        top_matches["Huong_Ten"] = top_matches["HuongNha"].map(lambda x: HUONG_NHA_MAP[x].split("(")[0].strip())
        top_matches["Gara"] = top_matches["CoGaraOto"].map({1: "Có", 0: "Không"})

        st.success(f"Tìm thấy **{len(filtered)}** căn nhà thỏa mãn tiêu chí! Dưới đây là **Top {len(top_matches)} lựa chọn tối ưu nhất**:")

        display_cols = [
            "DiemPhuHop", "DienTich_m2", "MatTien_m", "GiaNha_TyVND", "DonGia_Tr_m2",
            "KhuVuc_Ten", "DoRongDuong_m", "Huong_Ten", "SoTang", "SoPhongNgu",
            "TuoiNha_Nam", "Gara", "TietKiem_Ty"
        ]
        rename_cols = {
            "DiemPhuHop": "Điểm (/100)",
            "DienTich_m2": "Diện tích (m²)",
            "MatTien_m": "Mặt tiền (m)",
            "GiaNha_TyVND": "Giá bán (Tỷ)",
            "DonGia_Tr_m2": "Đơn giá (Tr/m²)",
            "KhuVuc_Ten": "Khu vực",
            "DoRongDuong_m": "Đường (m)",
            "Huong_Ten": "Hướng nhà",
            "SoTang": "Tầng",
            "SoPhongNgu": "Phòng ngủ",
            "TuoiNha_Nam": "Tuổi nhà",
            "Gara": "Gara",
            "TietKiem_Ty": "Dư ngân sách (Tỷ)"
        }
        st.dataframe(top_matches[display_cols].rename(columns=rename_cols), use_container_width=True, hide_index=True)

# =========================================================
# TAB 3: BÀI TOÁN TÀI CHÍNH & VAY TRẢ GÓP NGÂN HÀNG
# =========================================================
with tab3:
    st.subheader("🏦 Mô Phỏng Vay Mua Nhà & Kiểm Tra Sức Khỏe Tài Chính")
    st.caption(f"Tính toán dòng tiền trả nợ dựa trên giá trị căn nhà đang chọn: **{pred_price:.2f} Tỷ VNĐ** (Trọn gói: **{tong_chi_phi_lan_banh:.2f} Tỷ VNĐ**).")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        ty_le_von = st.slider("Tỷ lệ vốn tự có sẵn sàng (%)", 20, 90, 40, step=5)
    with m_col2:
        thoi_han_vay = st.slider("Thời hạn vay ngân hàng (năm)", 5, 30, 15, step=1)
    with m_col3:
        lai_suat_nam = st.number_input("Lãi suất vay trung bình (%/năm)", 4.0, 18.0, 8.5, step=0.2)
    with m_col4:
        thu_nhap_thang = st.number_input("Tổng thu nhập gia đình (Triệu/tháng)", 10.0, 500.0, 45.0, step=2.0)

    von_tu_co = round(pred_price * (ty_le_von / 100.0), 2)
    so_tien_vay = round(pred_price - von_tu_co, 2)
    so_thang = thoi_han_vay * 12
    r_thang = (lai_suat_nam / 100.0) / 12.0

    goc_hang_thang_trieu = (so_tien_vay * 1000.0) / so_thang
    lai_thang_dau_trieu = (so_tien_vay * 1000.0) * r_thang
    tra_thang_dau_trieu = goc_hang_thang_trieu + lai_thang_dau_trieu
    tong_lai_ty = ((so_tien_vay * r_thang) * (so_thang + 1)) / 2.0
    ty_le_tra_no_thu_nhap = (tra_thang_dau_trieu / thu_nhap_thang) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Vốn tự có cần chuẩn bị", f"{von_tu_co:.2f} Tỷ", delta=f"+{phi_sang_ten_uoc_tinh:.2f} Tỷ phí sang tên", delta_color="off")
    with k2:
        st.metric("Số tiền cần vay ngân hàng", f"{so_tien_vay:.2f} Tỷ", delta=f"Chiếm {100 - ty_le_von}% giá trị nhà", delta_color="off")
    with k3:
        st.metric("Tiền trả tháng đầu tiên", f"{tra_thang_dau_trieu:.1f} Tr/tháng", delta=f"Gốc {goc_hang_thang_trieu:.1f} Tr + Lãi {lai_thang_dau_trieu:.1f} Tr", delta_color="off")
    with k4:
        st.metric(
            "Áp lực trả nợ / Thu nhập",
            f"{ty_le_tra_no_thu_nhap:.1f}%",
            delta="An toàn (≤ 40% thu nhập)" if ty_le_tra_no_thu_nhap <= 40 else "Áp lực cao (> 40% thu nhập)",
            delta_color="normal" if ty_le_tra_no_thu_nhap <= 40 else "inverse"
        )

    if ty_le_tra_no_thu_nhap <= 40:
        st.success(f"✅ **Tỷ lệ tài chính an toàn ({ty_le_tra_no_thu_nhap:.1f}%):** Khoản trả nợ tháng đầu **{tra_thang_dau_trieu:.1f} Triệu** chiếm dưới 40% thu nhập, đảm bảo gia đình vẫn thoải mái chi tiêu sinh hoạt. Tổng tiền lãi ước tính trong {thoi_han_vay} năm là **{tong_lai_ty:.2f} Tỷ**.")
    else:
        st.warning(f"⚠️ **Cảnh báo áp lực tài chính ({ty_le_tra_no_thu_nhap:.1f}%):** Tiền trả tháng đầu chiếm quá 40% thu nhập. Bạn nên tăng vốn tự có lên mức **50–60%**, kéo dài thời hạn vay lên **20–25 năm** hoặc chọn căn diện tích nhỏ hơn.")

    years = np.arange(1, thoi_han_vay + 1)
    annual_principal, annual_interest = [], []
    remaining_loan = so_tien_vay * 1000.0
    for _ in years:
        y_p, y_i = 0.0, 0.0
        for _ in range(12):
            interest_m = remaining_loan * r_thang
            y_p += goc_hang_thang_trieu
            y_i += interest_m
            remaining_loan = max(0.0, remaining_loan - goc_hang_thang_trieu)
        annual_principal.append(round(y_p, 1))
        annual_interest.append(round(y_i, 1))

    df_loan = pd.DataFrame({
        "Năm": [f"Năm {y}" for y in years],
        "Tiền gốc trả (Triệu)": annual_principal,
        "Tiền lãi trả (Triệu)": annual_interest
    })
    fig_loan = px.bar(
        df_loan, x="Năm", y=["Tiền gốc trả (Triệu)", "Tiền lãi trả (Triệu)"],
        title="Biểu đồ nghĩa vụ trả Gốc & Lãi hàng năm (Dư nợ giảm dần)",
        color_discrete_sequence=["#2563EB", "#F59E0B"]
    )
    fig_loan.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_loan, use_container_width=True)

# =========================================================
# TAB 4: CHẤM ĐIỂM NGÔI NHÀ & CHECKLIST KHẢO SÁT THỰC TẾ
# =========================================================
with tab4:
    st.subheader("📋 Chấm Điểm Đánh Giá & Checklist Khi Đi Xem Nhà Thực Tế")
    col_sc1, col_sc2 = st.columns([1.1, 1])

    with col_sc1:
        st.markdown("#### 1. Biểu đồ đánh giá chất lượng toàn diện")
        diem_vi_tri = round((5 - khu_vuc) / 4 * 60 + min(do_rong_duong / 8, 1.0) * 40, 1)
        diem_phap_ly = 95.0 if phap_ly else 35.0
        diem_phong_thuy = 92.0 if huong_nha == 3 else (75.0 if huong_nha == 2 else 55.0)
        diem_xay_dung = round(max(40.0, 100.0 - tuoi_nha * 2.2) * 0.6 + (noi_that / 3) * 40, 1)
        diem_tai_chinh = 95.0 if chenh_lech_ngan_sach >= 0.3 else (75.0 if chenh_lech_ngan_sach >= 0 else 45.0)

        categories = ["Vị trí & Giao thông", "Pháp lý & Sổ hồng", "Hướng nhà & Thoáng khí", "Chất lượng & Nội thất", "Phù hợp tài chính"]
        scores = [diem_vi_tri, diem_phap_ly, diem_phong_thuy, diem_xay_dung, diem_tai_chinh]
        tong_diem_nha = round(float(np.mean(scores)), 1)

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
            title=f"Điểm tổng hợp ngôi nhà: {tong_diem_nha} / 100 điểm"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_sc2:
        st.markdown("#### 2. Checklist 8 bước bắt buộc khi đi xem nhà thực tế")
        st.caption("Tích chọn các hạng mục bạn đã trực tiếp kiểm tra tại hiện trường:")
        checks = [
            st.checkbox("1. Đối chiếu sổ hồng bản gốc, số tờ/số thửa và diện tích thực tế"),
            st.checkbox("2. Kiểm tra thông tin quy hoạch tại UBND quận/huyện (không dính lộ giới/treo)"),
            st.checkbox("3. Kiểm tra tình trạng tranh chấp, thế chấp ngân hàng hoặc đồng sở hữu"),
            st.checkbox("4. Quan sát tường, trần nhà vệ sinh & sân thượng xem có thấm dột, nứt kết cấu"),
            st.checkbox("5. Hỏi hàng xóm xung quanh về tình trạng ngập nước khi mưa lớn / triều cường"),
            st.checkbox("6. Kiểm tra áp lực nước sinh hoạt, hệ thống điện và thoát nước thải"),
            st.checkbox("7. Đo thực tế chiều rộng hẻm từ ngoài đường lớn vào đến cửa nhà"),
            st.checkbox("8. Khảo sát môi trường sống, tiếng ồn và an ninh khu vực vào buổi tối")
        ]
        done_count = sum(checks)
        st.progress(done_count / len(checks))
        st.write(f"**Tiến độ kiểm tra thực địa:** {done_count} / {len(checks)} hạng mục ({int(done_count/len(checks)*100)}%)")
        if done_count == len(checks):
            st.success("🎉 Bạn đã hoàn tất đầy đủ các bước thẩm định an toàn trước khi đặt cọc!")

# =========================================================
# TAB 5: SO SÁNH CÁC PHƯƠNG ÁN ĐÃ LƯU
# =========================================================
with tab5:
    st.subheader("⚖️ Bảng Đặt Lên Bàn Cân Các Ngôi Nhà Đang Cân Nhắc")
    st.write("Thay đổi thông số ở thanh bên trái và nhấn **'➕ Lưu phương án này vào bảng so sánh'** ở Tab 1 để đối chiếu nhiều căn nhà.")

    if len(st.session_state.saved_houses) == 0:
        st.info("Chưa có phương án nào được lưu. Hãy quay lại **Tab 1**, chọn thông số cho từng căn nhà và bấm nút lưu.")
    else:
        df_compare = pd.DataFrame(st.session_state.saved_houses)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns([3, 1])
        with c_left:
            fig_cmp = px.bar(
                df_compare, x="Tên phương án", y=["Giá dự đoán (Tỷ)", "Trọn gói (Tỷ)", "Dư/Thiếu NS (Tỷ)"],
                barmode="group", title="So sánh Giá bán, Chi phí trọn gói & Mức dư/vượt ngân sách"
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

# =========================================================
# TAB 6: NẠP & KIỂM TRA DỮ LIỆU EXCEL / CSV CỦA NHÓM
# =========================================================
with tab6:
    st.subheader("📁 Kiểm Tra & Phân Tích Tệp Dữ Liệu Excel / CSV")
    st.info(
        "💡 **Hướng dẫn chọn file trong bộ 3 file của Nhóm 5:**\n"
        "- Hãy tải lên **`Nhóm5_GiaNha_2.xlsx`** (tệp tổng 100% dữ liệu) để xem toàn bộ thông tin.\n"
        "- Không dùng riêng tệp `test` để huấn luyện vì chỉ chiếm 20% dữ liệu."
    )

    uploaded = st.file_uploader("Chọn tệp Excel (.xlsx) hoặc CSV (.csv)", type=["xlsx", "csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.success(f"Đã tải lên `{uploaded.name}`: **{len(df_up):,} dòng** × **{len(df_up.columns)} cột**.")
            st.dataframe(df_up.head(20), use_container_width=True)

            if {"Rank", "Score", "SubmissionCount"}.issubset(df_up.columns):
                st.warning(
                    "⚠️ Tệp này là **dữ liệu bảng xếp hạng cuộc thi Kaggle** (chứa `Rank`, `Score`, `SubmissionCount`), "
                    "không chứa đặc trưng căn nhà (`m²`, phòng ngủ, vị trí). App sẽ giữ nguyên bộ dữ liệu thị trường nhà ở ở các Tab 1–5 để bạn sử dụng."
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số đội tham gia", f"{len(df_up):,}")
                m2.metric("Điểm sai số trung vị (Score)", f"{df_up['Score'].median():.5f}")
                m3.metric("Số lần nộp bài trung bình", f"{df_up['SubmissionCount'].mean():.1f} lần")
        except Exception as e:
            st.error(f"Lỗi khi đọc tệp: {e}")
