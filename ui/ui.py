import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any

# =================== 1. 页面配置与美化样式 ===================

st.set_page_config(
    page_title="星尘AI视觉平台",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全新的CSS样式，更注重细节、间距和视觉层次
st.markdown("""
<style>
    /* 全局背景和字体 */
    .stApp {
        background-color: #f0f2f6;
    }

    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4e8;
    }
    .st-emotion-cache-16txt4v { /* 侧边栏标题 */
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1f36;
    }

    /* 顶级Tabs导航栏 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: transparent;
        border-bottom: 4px solid transparent;
        border-radius: 0;
        font-weight: 600;
        color: #6c757d;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #5a4fcf; /* 优雅的紫色主题 */
        color: #5a4fcf;
    }
    
    /* 自定义信息卡片 */
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    .info-card .icon {
        font-size: 2rem;
    }
    .info-card .title {
        font-weight: 600;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .info-card .value {
        font-weight: 700;
        color: #1a1f36;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =================== 2. API通信与数据处理 ===================

API_ENDPOINTS = {
    'HEALTH': '/api/face/health',
    'FACES': '/api/face/faces',
    'FACE_BY_SN': '/api/face/faces/{}',
    'RECOGNIZE': '/api/face/recognize',
    'STREAM': '/api/face/stream'
}

@st.cache_data(ttl=10)
def check_api_status(api_url: str) -> Tuple[bool, str]:
    try:
        response = requests.get(f"http://{api_url}{API_ENDPOINTS['HEALTH']}", timeout=3)
        return (True, "服务运行正常") if response.ok else (False, f"服务异常 (HTTP: {response.status_code})")
    except requests.RequestException:
        return False, "服务连接失败"

def api_request(method: str, endpoint: str, **kwargs) -> Tuple[bool, Any, str]:
    base_url = st.session_state.api_url
    full_url = f"http://{base_url}{endpoint}"
    try:
        response = requests.request(method, full_url, timeout=30, **kwargs)
        res_json = response.json()
        msg, data = res_json.get("msg", "无详细信息"), res_json.get("data")
        return response.ok, data, msg
    except requests.JSONDecodeError:
        return False, None, f"无法解析响应 (HTTP {response.status_code})"
    except requests.RequestException as e:
        return False, None, f"网络请求失败: {e}"

def refresh_data(show_toast=True):
    """获取最新人脸库数据并更新session_state"""
    success, data, msg = api_request('GET', API_ENDPOINTS['FACES'])
    if success and data:
        all_faces = data.get('faces', [])
        st.session_state.all_faces = all_faces
        unique_sns = {face['sn'] for face in all_faces}
        st.session_state.people_list = sorted(list(unique_sns))
        st.session_state.person_count = len(unique_sns)
        if show_toast: st.toast("数据已刷新!", icon="🔄")
    else:
        st.session_state.all_faces, st.session_state.people_list, st.session_state.person_count = [], [], 0
        st.error(f"数据加载失败: {msg}")

# =================== 3. UI渲染模块 ===================

def render_sidebar():
    with st.sidebar:
        st.title("💫 星尘AI视觉平台")
        st.caption("v4.0.0 - 企业版")
        st.session_state.api_url = st.text_input("后端服务地址", value="172.16.2.34:12010")
        is_connected, status_msg = check_api_status(st.session_state.api_url)
        st.info(f"**API状态:** {status_msg}", icon="📡" if is_connected else "🔌")
        st.divider()
        if st.button("🔄 强制刷新全站数据", use_container_width=True):
            refresh_data()
        st.info("© 2025-2026 版权所有")

@st.dialog("➕ 注册新人员")
def register_person_dialog():
    with st.form("new_person_form"):
        st.subheader("新人员信息录入")
        name = st.text_input("姓名")
        sn = st.text_input("唯一编号(SN)")
        image_file = st.file_uploader("上传人脸照片")
        if st.form_submit_button("确认注册", type="primary"):
            if name and sn and image_file:
                params = {'name': name, 'sn': sn}
                files = {'image_file': (image_file.name, image_file.getvalue())}
                success, _, msg = api_request('POST', API_ENDPOINTS['FACES'], params=params, files=files)
                if success:
                    st.toast(f"注册成功: {msg}", icon="🎉")
                    refresh_data(show_toast=False)
                else:
                    st.error(f"注册失败: {msg}")
            else:
                st.warning("所有字段均为必填项。")


def render_dashboard_page():
    st.header("📊 仪表盘")
    col1, col2 = st.columns(2)
    with col1:
        st.html(f"""
        <div class="info-card">
            <div class="icon">👥</div>
            <div class="title">人脸库人员总数</div>
            <div class="value">{st.session_state.get('person_count', 0)}</div>
        </div>
        """)
    with col2:
        api_status, api_color = ("在线", "#28a745") if check_api_status(st.session_state.api_url)[0] else ("离线", "#dc3545")
        st.html(f"""
        <div class="info-card">
            <div class="icon">📡</div>
            <div class="title">API服务状态</div>
            <div class="value" style="color:{api_color};">{api_status}</div>
        </div>
        """)
    st.divider()

    st.header("🧐 快速识别")
    with st.container(border=True):
        uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            col_img, col_res = st.columns(2)
            col_img.image(uploaded_file, caption="待识别图片预览")
            if st.button("开始识别", use_container_width=True, type="primary"):
                files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                success, data, msg = api_request('POST', API_ENDPOINTS['RECOGNIZE'], files=files)
                with col_res:
                    if success and data:
                        st.success(f"识别成功！找到 {len(data)} 个匹配项。")
                        for result in data:
                            st.info(f"**姓名:** {result.get('name')} | **SN:** {result.get('sn')} | **距离:** {result.get('distance', 0):.4f}")
                    else:
                        st.error(f"识别失败: {msg}")

def render_management_page():
    st.header("🗂️ 人脸库管理中心")
    
    # 顶层操作区
    if st.button("➕ 注册新人员", type="primary"):
        register_person_dialog()
    
    management_tabs = st.tabs(["**浏览与操作**", "**数据看板**"])

    with management_tabs[0]: # 浏览与操作
        st.markdown("---")
        list_col, action_col = st.columns([2, 3])

        with list_col:
            st.subheader("👥 人员列表")
            people_list = st.session_state.get('people_list', [])
            if not people_list:
                st.info("人脸库为空，请点击上方按钮注册人员。")
            else:
                for person_sn in people_list:
                    is_selected = (st.session_state.get('selected_sn') == person_sn)
                    card_bg = "#eef2ff" if is_selected else "#ffffff"
                    with st.container(border=True):
                         st.markdown(f"<div style='background-color:{card_bg}; padding: 10px; border-radius: 8px;'>", unsafe_allow_html=True)
                         c1, c2 = st.columns([3, 2])
                         name = next((f['name'] for f in st.session_state.all_faces if f['sn'] == person_sn), "N/A")
                         c1.markdown(f"**{name}**<br><small>SN: {person_sn}</small>", unsafe_allow_html=True)
                         if c2.button("选择", key=f"manage_{person_sn}", use_container_width=True):
                             st.session_state.selected_sn = person_sn
                             st.rerun()
                         st.markdown("</div>", unsafe_allow_html=True)

        with action_col:
            st.subheader("⚙️ 操作面板")
            selected_sn = st.session_state.get('selected_sn')
            if not selected_sn:
                st.info("⬅️ 请从左侧列表中选择一个人员进行管理。")
            else:
                name = next((p['name'] for p in st.session_state.all_faces if p['sn'] == selected_sn), "N/A")
                with st.container(border=True):
                    st.markdown(f"#### 正在管理: **{name}** (`{selected_sn}`)")
                    action_tabs = st.tabs(["✏️ 更新信息", "🗑️ 删除人员"])
                    with action_tabs[0]:
                        with st.form(f"update_form_{selected_sn}"):
                            new_name = st.text_input("新姓名", value=name)
                            if st.form_submit_button("保存更改", type="primary"):
                                if new_name and new_name != name:
                                    endpoint = API_ENDPOINTS['FACE_BY_SN'].format(selected_sn)
                                    success, _, msg = api_request('PUT', endpoint, json={"name": new_name})
                                    if success:
                                        st.toast(f"更新成功: {msg}", icon="✅"); refresh_data(False); st.rerun()
                                    else: st.error(f"更新失败: {msg}")
                    with action_tabs[1]:
                        st.error(f"警告：此操作将永久删除 **{name}** 的所有记录！")
                        if st.button(f"我确认要删除 {name}", type="primary"):
                            endpoint = API_ENDPOINTS['FACE_BY_SN'].format(selected_sn)
                            success, _, msg = api_request('DELETE', endpoint)
                            if success:
                                st.toast(f"'{name}' 已删除", icon="🗑️"); st.session_state.selected_sn = None; refresh_data(False); st.rerun()
                            else: st.error(f"删除失败: {msg}")
    
    with management_tabs[1]: # 数据看板
        st.subheader("📈 数据看板")
        if st.session_state.get('all_faces'):
            df = pd.DataFrame(st.session_state.all_faces)
            df['registration_time'] = pd.to_datetime(df['registration_time'])
            df['registration_date'] = df['registration_time'].dt.date
            daily_regs = df.groupby('registration_date').size().reset_index(name='counts')
            st.bar_chart(daily_regs, x='registration_date', y='counts')
        else:
            st.info("暂无数据可供分析。")

def render_monitoring_page():
    st.header("🛰️ 实时视频监测")
    with st.container(border=True):
        video_source = st.text_input("视频源地址", value="rtmp://ns8.indexforce.com/home/mystream")
        col1, col2 = st.columns(2)
        if col1.button("▶️ 开启监测", use_container_width=True, type="primary"):
            st.session_state.stream_url = f"http://{st.session_state.api_url}{API_ENDPOINTS['STREAM']}?source={video_source}"
        if col2.button("⏹️ 停止监测", use_container_width=True):
            st.session_state.stream_url = None
        if st.session_state.get("stream_url"):
            st.image(st.session_state.stream_url, caption=f"实时视频流 | 源: {video_source}")
        else:
            st.info("视频流已停止或未开启。")

# =================== 4. 主程序入口 ===================
def main():
    render_sidebar()
    st.title("星尘AI视觉平台")
    if 'all_faces' not in st.session_state:
        refresh_data(show_toast=False)

    tab1, tab2, tab3 = st.tabs(["**📊 仪表盘**", "**🗂️ 人脸库管理**", "**🛰️ 实时监测**"])
    with tab1:
        render_dashboard_page()
    with tab2:
        render_management_page()
    with tab3:
        render_monitoring_page()

if __name__ == "__main__":
    main()