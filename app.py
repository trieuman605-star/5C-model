import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import io

# ==========================================
# 1) PAGE CONFIGURATION (Phải là lệnh Streamlit đầu tiên)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Đánh giá Rủi ro Tín dụng - Mô hình 5C",
    page_icon="💳"
)

# ==========================================
# 2) CACHED FUNCTIONS & UTILITIES
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ file upload và lưu cache ứng dụng"""
    if file_name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))

# Định nghĩa danh sách các biến đầu vào chính xác theo mô hình 5C trong Notebook
FEATURES = [
    'TC1', 'TC2', 'TC3', 'TC4', 'TC5',  # Tư cách (Character)
    'NL1', 'NL2', 'NL3', 'NL4',         # Năng lực (Capacity)
    'DK1', 'DK2', 'DK3', 'DK4', 'DK5',  # Điều kiện (Conditions)
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',  # Vốn (Capital)
    'TS1', 'TS2', 'TS3', 'TS4'          # Tài sản đảm bảo (Collateral)
]
TARGET = 'PD' # Biến mục tiêu: 0 - Không rủi ro, 1 - Có rủi ro

# ==========================================
# 3) SIDEBAR - VÙNG CẤU HÌNH & ĐIỀU KHIỂN
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Thành phần tải dữ liệu mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp chứa các cột tiêu chí định giá từ TC1 đến TS4 và biến mục tiêu PD."
    )
    
    st.divider()
    st.subheader("🤖 Tham số mô hình Logistic Regression")
    
    # Cho phép tinh chỉnh siêu tham số mô hình dựa trên thực tế cấu hình học máy
    c_value = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Giá trị nhỏ hơn tăng cường hiệu ứng điều hòa (regularization)")
    random_state = st.number_input("Random State", value=23, step=1, help="Khớp định danh phân tách dữ liệu giống cấu hình gốc của Notebook")
    test_size = st.slider("Tỷ lệ tập kiểm định (Test size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Tỷ lệ chia tập test dữ liệu đầu vào")
    
    st.divider()
    # Điểm kích hoạt huấn luyện duy nhất của ứng dụng
    btn_train = st.button("🎯 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# 4) HEADER - VÙNG ĐỊNH HƯỚNG ỨNG DỤNG
# ==========================================
st.title("📊 Ứng dụng Học Máy Đánh giá Rủi ro Khách hàng (Mô hình 5C)")
st.caption("Ứng dụng phân tích dữ liệu xếp hạng tín dụng doanh nghiệp/cá nhân dựa trên 5 khía cạnh cốt lõi: Tư cách, Năng lực, Điều kiện kinh doanh, Vốn và Tài sản đảm bảo.")

# Kiểm tra trạng thái dữ liệu đầu vào (Xử lý trạng thái rỗng)
if uploaded_file is None:
    st.info("👋 Vui lòng tải dữ liệu mẫu từ thanh bên (Sidebar) để kích hoạt toàn bộ tính năng phân tích.")
    st.stop()

# Đọc dữ liệu sau khi đảm bảo tệp đã được tải lên thành công
df = load_data(uploaded_file.getvalue(), uploaded_file.name)
st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}`")
st.divider()

# ==========================================
# 5) KHỐI XỬ LÝ & HUẤN LUYỆN MÔ HÌNH (LƯU SESSION STATE)
# ==========================================
if btn_train:
    # Kiểm tra tính hợp lệ của cấu trúc cột dữ liệu đầu vào
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {missing_cols}")
    else:
        with st.spinner("Mô hình đang học dữ liệu..."):
            # Trích xuất biến độc lập X và biến mục tiêu y
            X = df[FEATURES]
            y = df[TARGET]
            
            # Phân tách tập dữ liệu Train/Test theo cấu hình thu thập từ Sidebar
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Khởi tạo thuật toán và khớp dữ liệu huấn luyện
            model = LogisticRegression(C=c_value, max_iter=1000, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Tính toán các chỉ số đánh giá và lưu trữ bền vững vào session_state
            y_pred = model.predict(X_test)
            acc_score = model.score(X_test, y_test)
            cm = confusion_matrix(y_test, y_pred)
            
            st.session_state['trained_model'] = model
            st.session_state['accuracy'] = acc_score
            st.session_state['confusion_matrix'] = cm
            st.session_state['features_list'] = FEATURES
            st.session_state['summary_df'] = df.copy()
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab bên dưới để xem kết quả.")

# ==========================================
# 6) HỆ THỐNG PHÂN VÙNG CHỨC NĂNG - TABS LAYOUT
# ==========================================
tab_overview, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📋 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🏆 Kết quả & Kiểm định",
"🔮 Sử dụng mô hình"
])

# ------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab_overview:
    st.subheader("Phân tích Thống kê Mô tả Thô")
    
    # Hiển thị thông số kích thước cơ bản của tệp
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số dòng (Quan sát)", df.shape[0])
    with col_m2:
        st.metric("Số lượng cột", df.shape[1])
    with col_m3:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
        
    st.write("##### 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("##### Thống kê mô tả các biến đặc trưng trong mô hình máy học:")
    # Chỉ mô tả phân phối các cột tham gia vào cấu trúc mô hình
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# ------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab_viz:
    st.subheader("Trực quan hóa Phân phối Đặc trưng")
    
    # Bộ chọn động các biến cần hiển thị nếu danh sách biến quá dài
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng đầu vào muốn quan sát cùng biến mục tiêu:",
        options=FEATURES,
        default=FEATURES[:3],
        max_selections=3
    )
    
    # Lưới bố cục hiển thị biểu đồ 2x2 cân đối
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Biểu đồ biến mục tiêu PD bắt buộc hiển thị đầu tiên
    with row1_col1:
        if TARGET in df.columns:
            target_counts = df[TARGET].value_counts().reset_index()
            target_counts.columns = ['Trạng thái Rủi ro', 'Số lượng']
            target_counts['Trạng thái Rủi ro'] = target_counts['Trạng thái Rủi ro'].map({0: '0 - Không rủi ro', 1: '1 - Có rủi ro'})
            fig_target = px.bar(target_counts, x='Trạng thái Rủi ro', y='Số lượng', 
                                title="Phân phối Biến Mục Tiêu (PD)", color='Trạng thái Rủi ro',
                                color_discrete_map={'0 - Không rủi ro': '#2ecc71', '1 - Có rủi ro': '#e74c3c'})
            st.plotly_chart(fig_target, use_container_width=True)
            
    # 2. Biểu đồ cho các đặc trưng tùy chọn tiếp theo
    grid_positions = [row1_col2, row2_col1, row2_col2]
    for idx, feat in enumerate(selected_features):
        if idx < len(grid_positions):
            with grid_positions[idx]:
                fig_feat = px.histogram(df, x=feat, color=TARGET, barmode='group',
                                        title=f"Phân phối tần suất điểm của tiêu chí {feat}",
labels={TARGET: 'Trạng thái PD'},
                                        color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab_metrics:
    st.subheader("Hiệu suất định lượng của thuật toán AI")
    
    # Điều phối luồng trạng thái huấn luyện
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng bấm nút **'Huấn luyện mô hình'** tại thanh bên trái để xem báo cáo đánh giá thuật toán.")
    else:
        acc = st.session_state['accuracy']
        cm = st.session_state['confusion_matrix']
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Độ chính xác tổng thể (Accuracy Score)", value=f"{acc * 100:.2f}%")
            st.write("---")
            st.write("**Bảng giải nghĩa phân loại:**")
            st.markdown("- **0**: Khách hàng an toàn\n- **1**: Khách hàng rủi ro tiềm ẩn")
            
        with col_res2:
            st.write("##### Ma trận nhầm lẫn (Confusion Matrix):")
            # Trình bày Ma trận nhầm lẫn dưới dạng cấu trúc bảng minh bạch nhãn phân loại công việc
            cm_df = pd.DataFrame(
                cm, 
                index=['Thực tế: Không rủi ro (0)', 'Thực tế: Có rủi ro (1)'],
                columns=['Dự báo: Không rủi ro (0)', 'Dự báo: Có rủi ro (1)']
            )
            st.dataframe(cm_df, use_container_width=True)

# ------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO TÁC NGHIỆP
# ------------------------------------------
with tab_inference:
    st.subheader("Vận hành mô hình kiểm tra hồ sơ mới")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng huấn luyện mô hình thành công trước khi khai thác tính năng dự báo hồ sơ.")
    else:
        model = st.session_state['trained_model']
        
        # Chọn chế độ nhập liệu theo quy tắc
        mode = st.radio("Chọn phương thức nhập đầu vào:", ["Nhập tay hồ sơ đơn lẻ", "Dự báo hàng loạt theo danh sách Tệp"], horizontal=True)
        
        # CHẾ ĐỘ 1: NHẬP LIỆU FORM TRỰC TIẾP
        if mode == "Nhập tay hồ sơ đơn lẻ":
            st.write("##### Điền điểm đánh giá các tiêu chí chuyên gia (Thang điểm từ 1 - 5):")
            
            with st.form(key='single_inference_form'):
                # Gom cụm bố cục nhập liệu theo cấu trúc 5C giúp tăng tính công thái học
                st.markdown("**1. Tư cách người vay (Character - TC)**")
                c1, c2, c3, c4, c5 = st.columns(5)
                tc1 = c1.slider("TC1", 1, 5, 4, help="Tư cách khách hàng tiêu chí số 1")
                tc2 = c2.slider("TC2", 1, 5, 5, help="Tư cách khách hàng tiêu chí số 2")
                tc3 = c3.slider("TC3", 1, 5, 5, help="Tư cách khách hàng tiêu chí số 3")
                tc4 = c4.slider("TC4", 1, 5, 5, help="Tư cách khách hàng tiêu chí số 4")
                tc5 = c5.slider("TC5", 1, 5, 3, help="Tư cách khách hàng tiêu chí số 5")
                
                st.markdown("**2. Năng lực tài chính (Capacity - NL)**")
                nl_c1, nl_c2, nl_c3, nl_c4 = st.columns(4)
                nl1 = nl_c1.slider("NL1", 1, 5, 4, help="Năng lực hoạt động tiêu chí số 1")
                nl2 = nl_c2.slider("NL2", 1, 5, 2, help="Năng lực hoạt động tiêu chí số 2")
                nl3 = nl_c3.slider("NL3", 1, 5, 4, help="Năng lực hoạt động tiêu chí số 3")
                nl4 = nl_c4.slider("NL4", 1, 5, 4, help="Năng lực hoạt động tiêu chí số 4")
                
                st.markdown("**3. Điều kiện vĩ mô & Kinh doanh (Conditions - DK)**")
                dk_c1, dk_c2, dk_c3, dk_c4, dk_c5 = st.columns(5)
                dk1 = dk_c1.slider("DK1", 1, 5, 4, help="Điều kiện môi trường kinh doanh số 1")
                dk2 = dk_c2.slider("DK2", 1, 5, 5, help="Điều kiện môi trường kinh doanh số 2")
                dk3 = dk_c3.slider("DK3", 1, 5, 3, help="Điều kiện môi trường kinh doanh số 3")
                dk4 = dk_c4.slider("DK4", 1, 5, 3, help="Điều kiện môi trường kinh doanh số 4")
                dk5 = dk_c5.slider("DK5", 1, 5, 5, help="Điều kiện môi trường kinh doanh số 5")
                
                st.markdown("**4. Tiêu chí Vốn tự có (Capital - V)**")
                v_c1, v_c2, v_c3, v_c4, v_c5, v_c6 = st.columns(6)
                v1 = v_c1.slider("V1", 1, 5, 5, help="Cơ cấu vốn tiêu chí số 1")
                v2 = v_c2.slider("V2", 1, 5, 4, help="Cơ cấu vốn tiêu chí số 2")
                v3 = v_c3.slider("V3", 1, 5, 3, help="Cơ cấu vốn tiêu chí số 3")
                v4 = v_c4.slider("V4", 1, 5, 2, help="Cơ cấu vốn tiêu chí số 4")
                v5 = v_c5.slider("V5", 1, 5, 3, help="Cơ cấu vốn tiêu chí số 5")
                v6 = v_c6.slider("V6", 1, 5, 4, help="Cơ cấu vốn tiêu chí số 6")
                
                st.markdown("**5. Tài sản bảo đảm (Collateral - TS)**")
                ts_c1, ts_c2, ts_c3, ts_c4 = st.columns(4)
                ts1 = ts_c1.slider("TS1", 1, 5, 5, help="Giá trị pháp lý tài sản số 1")
                ts2 = ts_c2.slider("TS2", 1, 5, 4, help="Giá trị pháp lý tài sản số 2")
                ts3 = ts_c3.slider("TS3", 1, 5, 4, help="Giá trị pháp lý tài sản số 3")
                ts4 = ts_c4.slider("TS4", 1, 5, 3, help="Giá trị pháp lý tài sản số 4")
btn_predict = st.form_submit_button("🚀 Đánh giá rủi ro hồ sơ")
                
            if btn_predict:
                # Tổ chức vectơ đầu vào khớp hoàn toàn dạng mảng 2 chiều theo cấu hình huấn luyện gốc
                X_new = [[
                    tc1, tc2, tc3, tc4, tc5,
                    nl1, nl2, nl3, nl4,
                    dk1, dk2, dk3, dk4, dk5,
                    v1, v2, v3, v4, v5, v6,
                    ts1, ts2, ts3, ts4
                ]]
                
                pred = model.predict(X_new)[0]
                probs = model.predict_proba(X_new)[0]
                
                st.write("---")
                st.subheader("🎯 Kết quả phân tích quyết định")
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    if pred == 1:
                        st.error("🚨 KẾT QUẢ: KHÁCH HÀNG CÓ NGUY CƠ RỦI RO CAO")
                    else:
                        st.success("✅ KẾT QUẢ: KHÁCH HÀNG AN TOÀN")
                        
                with col_p2:
                    st.metric("Xác suất KHÔNG rủi ro", f"{probs[0]*100:.2f}%")
                    st.metric("Xác suất CÓ rủi ro", f"{probs[1]*100:.2f}%")

        # CHẾ ĐỘ 2: TẢI TỆP HÀNG LOẠT
        elif mode == "Dự báo hàng loạt theo danh sách Tệp":
            st.write("##### Tải lên tệp chứa thông tin danh sách hồ sơ mới cần phân loại:")
            st.info("📌 Ghi chú: Tệp tải lên cần chứa đủ các cột thuộc tính định dạng số từ `TC1` đến `TS4` tương đương cấu trúc tệp mẫu huấn luyện gốc.")
            
            batch_file = st.file_uploader("Tải lên danh sách kiểm tra (.csv, .xlsx)", type=["csv", "xlsx"], key="batch_upload")
            
            if batch_file:
                # Đọc tệp dữ liệu dự báo tác nghiệp hàng loạt
                if batch_file.name.endswith('.csv'):
                    batch_df = pd.read_csv(batch_file)
                else:
                    batch_df = pd.read_excel(batch_file)
                    
                # Kiểm tra tính khớp Schema biến đầu vào
                missing_batch_cols = [col for col in FEATURES if col not in batch_df.columns]
                
                if missing_batch_cols:
                    st.error(f"❌ Không thể thực thi. Tệp tải lên thiếu các trường thông tin kiểm tra sau: {missing_batch_cols}")
                else:
                    # Dự báo nhãn lớp và xác suất trực tiếp trên cấu trúc mảng của DataFrame mới
                    X_batch = batch_df[FEATURES]
                    batch_preds = model.predict(X_batch)
                    batch_probs = model.predict_proba(X_batch)[:, 1]
# Gán ngược kết quả vào DataFrame hiển thị đầu ra cho người dùng tác nghiệp
                    res_df = batch_df.copy()
                    res_df['Dự_báo_Nhãn_PD'] = batch_preds
                    res_df['Xác_suất_Rủi_ro_Có_Mô_Hình_5C (%)'] = np.round(batch_probs * 100, 2)
                    
                    st.write("##### Bảng kết quả chấm điểm tự động từ hệ thống AI:")
                    st.dataframe(res_df, use_container_width=True)
                    
                    # Trích xuất dữ liệu trả về máy trạm người dùng tác nghiệp dạng CSV định dạng chuẩn tiếng Việt Excel
                    csv_buffer = io.StringIO()
                    res_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 Tải xuống bảng kết quả định dạng CSV",
                        data=csv_buffer.getvalue(),
                        file_name="Ket_Qua_Danh_Gia_5C_Tidụng.csv",
                        mime="text/csv"
                    )
