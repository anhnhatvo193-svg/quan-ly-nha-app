import streamlit as st
import pandas as pd
import plotly.express as px

# Cấu hình trang
st.set_page_config(
    page_title="Dashboard Phân Tích Kết Quả Nhóm 5 - Giá Nhà",
    page_icon="📊",
    layout="wide"
)

# Hàm load dữ liệu (có caching để tối ưu tốc độ)
@st.cache_data
def load_data():
    df_all = pd.read_excel("Nhóm5_GiaNha.xlsx")
    df_train = pd.read_excel("Nhóm5 _GiaNha_train.xlsx")
    df_test = pd.read_excel("Nhóm5_GiaNha_test.xlsx")
    
    # Ép kiểu dữ liệu thời gian
    for df in [df_all, df_train, df_test]:
        df['LastSubmissionDate'] = pd.to_datetime(df['LastSubmissionDate'])
        
    return df_all, df_train, df_test

df_all, df_train, df_test = load_data()

# Tiêu đề ứng dụng
st.title("📊 Dashboard Phân Tích & Tra Cứu Kết Quả Dự Đoán Giá Nhà")
st.markdown("Ứng dụng trực quan hóa bảng xếp hạng, phân bố điểm số và lịch sử nộp bài của các đội thi.")

# Sidebar chọn tập dữ liệu & bộ lọc
st.sidebar.header("⚙️ Bộ Lọc & Tùy Chọn")
dataset_option = st.sidebar.selectbox(
    "Chọn tập dữ liệu hiển thị:",
    ["Tổng hợp (All)", "Tập Train", "Tập Test"]
)

if dataset_option == "Tổng hợp (All)":
    current_df = df_all
elif dataset_option == "Tập Train":
    current_df = df_train
else:
    current_df = df_test

# Lọc theo khoảng điểm (Score)
min_score = float(current_df['Score'].min())
max_score = float(current_df['Score'].max())
score_range = st.sidebar.slider(
    "Lọc theo điểm số (Score):",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score)
)

# Lọc theo từ khóa tên Đội / Thành viên
search_term = st.sidebar.text_input("🔍 Tìm tên Đội hoặc Thành viên:")

# Áp dụng bộ lọc
filtered_df = current_df[
    (current_df['Score'] >= score_range[0]) & 
    (current_df['Score'] <= score_range[1])
]

if search_term:
    filtered_df = filtered_df[
        filtered_df['TeamName'].str.contains(search_term, case=False, na=False) |
        filtered_df['TeamMemberUserNames'].str.contains(search_term, case=False, na=False)
    ]

# 1. Các chỉ số Tổng quan (KPIs)
st.subheader(f"📌 Tổng quan chỉ số - {dataset_option}")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Tổng số đội", f"{len(filtered_df):,} đội")
with col2:
    st.metric("Điểm tốt nhất (Min Score)", f"{filtered_df['Score'].min():.5f}")
with col3:
    st.metric("Điểm trung bình", f"{filtered_df['Score'].mean():.4f}")
with col4:
    st.metric("Lượt submit trung bình", f"{filtered_df['SubmissionCount'].mean():.1f} lần")

st.divider()

# 2. Biểu đồ phân tích
tab1, tab2, tab3 = st.tabs(["🏆 Top Đội Dẫn Đầu", "📈 Phân Bố Điểm & Submit", "📊 So Sánh Train vs Test"])

with tab1:
    st.markdown("### Top 15 đội xuất sắc nhất")
    top15 = filtered_df.sort_values(by="Rank").head(15)
    fig_top = px.bar(
        top15,
        x='TeamName',
        y='Score',
        color='SubmissionCount',
        title="Top 15 Đội thi (Xếp theo Rank)",
        labels={'TeamName': 'Tên đội', 'Score': 'Điểm số (Càng thấp càng tốt)', 'SubmissionCount': 'Số lần Submit'},
        text_auto='.4f'
    )
    fig_top.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_top, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Phân bố điểm số (Score Distribution)")
        fig_hist = px.histogram(
            filtered_df, 
            x='Score', 
            nbins=40, 
            title="Mật độ phân bố Score",
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        st.markdown("### Tương quan giữa Số lần Submit & Điểm số")
        fig_scatter = px.scatter(
            filtered_df,
            x='SubmissionCount',
            y='Score',
            hover_data=['TeamName', 'Rank'],
            title="Số lần Submit vs Score",
            color_discrete_sequence=['#EF553B']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### So sánh chỉ số giữa tập Train và Test")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Tập Train:**", df_train[['Score', 'SubmissionCount']].describe())
    with col_t2:
        st.write("**Tập Test:**", df_test[['Score', 'SubmissionCount']].describe())

st.divider()

# 3. Bảng dữ liệu chi tiết & Tải về
st.subheader("📋 Bảng dữ liệu chi tiết")
st.dataframe(filtered_df, use_container_width=True)

# Nút tải xuống dữ liệu dạng CSV
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Tải dữ liệu đã lọc (CSV)",
    data=csv_data,
    file_name=f"du_lieu_{dataset_option.lower()}.csv",
    mime="text/csv"
)