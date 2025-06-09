import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any, Dict, List
from pathlib import Path
import time
import json
from datetime import datetime

# ==============================================================================
# 1. 页面配置与美化 (Page Config & Styling)
# ==============================================================================

st.set_page_config(
    page_title="星尘AI视觉平台",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全新设计的CSS样式 ---
st.markdown("""
<style>
    /* --- 全局与字体 --- */
    .stApp { background-color: #f0f2f6; }
    h1, h2, h3 { font-weight: 700; color: #1a1f36; }

    /* --- 侧边栏 --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4e8;
    }

    /* --- 使用 st.radio 模拟 Tabs 的核心CSS --- */
    [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 1.5rem;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0;
        margin-bottom: 1.5rem;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label {
        height: 50px;
        padding: 0 1rem;
        background-color: transparent;
        border-bottom: 4px solid transparent;
        border-radius: 0;
        font-weight: 600;
        color: #6c757d;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        margin: 0;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        border-bottom: 4px solid #4f46e5;
        color: #4f46e5;
    }

    /* --- 自定义信息卡片 --- */
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .info-card:hover { transform: translateY(-5px); }
    .info-card .icon { font-size: 2.5rem; }
    .info-card .title { font-weight: 600; color: #6c757d; font-size: 1rem; margin-top: 10px; }
    .info-card .value { font-weight: 700; color: #1a1f36; font-size: 2rem; }

    /* --- 其他美化 --- */
    .stButton>button { border-radius: 8px; font-weight: 600; }
    [data-testid="stExpander"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. 会话状态管理 (Session State Management)
# ==============================================================================

def initialize_session_state():
    """初始化应用所需的全部会话状态。"""
    defaults = {
        "api_url": "172.16.4.152:12010",  # 默认指向后端的12010端口
        "api_status": (False, "尚未连接"),
        "faces_data": None,
        "show_register_dialog": False,
        "active_page": "仪表盘",
        "viewing_stream_info": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# 3. API通信与数据处理 (API Communication & Data Handling)
# ==============================================================================

API_ENDPOINTS = {
    'HEALTH': '/api/face/health',
    'FACES': '/api/face/faces',
    'FACE_BY_SN': '/api/face/faces/{}',
    'RECOGNIZE': '/api/face/recognize',
    'STREAMS_START': '/api/face/streams/start',
    'STREAMS_STOP': '/api/face/streams/stop/{}',
    'STREAMS_LIST': '/api/face/streams',
}


@st.cache_data(ttl=10)
def check_api_status(api_url: str) -> Tuple[bool, str]:
    """检查后端API的健康状况。"""
    try:
        url = f"http://{api_url}{API_ENDPOINTS['HEALTH']}"
        response = requests.get(url, timeout=3)
        if response.ok:
            return True, response.json().get("data", {}).get("message", "服务运行正常")
        return False, f"服务异常 (HTTP: {response.status_code})"
    except requests.RequestException:
        return False, "服务连接失败"


def parse_error_message(response: requests.Response) -> str:
    """智能解析后端的错误信息。"""
    try:
        res_json = response.json()
        if "detail" in res_json:
            detail = res_json["detail"]
            if isinstance(detail, list) and detail:
                first_error = detail[0]
                field_location = " → ".join(map(str, first_error.get("loc", [])))
                message = first_error.get("msg", "未知验证错误")
                field_location = field_location.replace("body", "请求体").replace("query", "查询参数")
                return f"字段 '{field_location}' 无效: {message}"
            elif isinstance(detail, str):
                return detail
        if "msg" in res_json:
            return res_json["msg"]
        return response.text
    except json.JSONDecodeError:
        return f"无法解析响应 (HTTP {response.status_code}): {response.text}"


def api_request(method: str, endpoint: str, **kwargs) -> Tuple[bool, Any, str]:
    """统一的API请求函数。"""
    full_url = f"http://{st.session_state.api_url}{endpoint}"
    try:
        response = requests.request(method, full_url, timeout=30, **kwargs)
        if response.ok:
            if response.status_code == 204 or not response.content:
                return True, None, "操作成功"
            res_json = response.json()
            # 确保即使data为None也返回成功状态
            return True, res_json.get("data"), res_json.get("msg", "操作成功")
        else:
            error_message = parse_error_message(response)
            return False, None, error_message
    except requests.RequestException as e:
        return False, None, f"网络请求失败: {e}"


def refresh_all_data():
    """从API获取最新的人脸库数据。"""
    with st.spinner("正在从服务器同步最新数据..."):
        success, data, msg = api_request('GET', API_ENDPOINTS['FACES'])
        if success and data:
            all_faces = data.get('faces', [])
            unique_sns = sorted(list({face['sn'] for face in all_faces}))
            st.session_state.faces_data = {
                "count": data.get('count', 0),
                "faces": all_faces,
                "unique_sns": unique_sns
            }
            st.toast("人脸库数据已同步!", icon="🔄")
        else:
            st.session_state.faces_data = {"count": 0, "faces": [], "unique_sns": []}
            st.error(f"人脸库数据加载失败: {msg}")


def convert_path_to_url(server_path: str) -> str:
    """将后端返回的文件路径智能地转换为可访问的URL。"""
    if not server_path or not isinstance(server_path, str):
        return "https://via.placeholder.com/150?text=No+Path"
    p = Path(server_path).as_posix()
    if 'data/' in p:
        relative_path = p.split('data/', 1)[1]
        return f"http://{st.session_state.api_url}/data/{relative_path}"
    return f"https://via.placeholder.com/150?text=Path+Error"


def format_datetime_human(dt_str: str) -> str:
    """将ISO格式的日期时间字符串转换为人性化的格式"""
    if not dt_str:
        return "永久"
    try:
        # 移除可能存在的微秒部分的小数点后的多余数字
        if '.' in dt_str:
            dt_str = dt_str.split('.')[0]
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        return dt_obj.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "N/A"


# ==============================================================================
# 4. UI渲染模块 (UI Rendering Modules)
# ==============================================================================

def render_sidebar():
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("💫 星尘AI视觉平台")
        st.caption("v8.0.0 - 体验优化版")

        st.session_state.api_url = st.text_input("后端服务地址", value=st.session_state.api_url)

        is_connected, status_msg = check_api_status(st.session_state.api_url)
        st.session_state.api_status = (is_connected, status_msg)
        status_icon = "✅" if is_connected else "❌"
        st.info(f"**API状态:** {status_msg}", icon=status_icon)

        st.divider()
        if st.button("🔄 强制刷新全站数据", use_container_width=True):
            refresh_all_data()

        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.info("© 2025-2026 版权所有")


@st.dialog("➕ 注册新人员", width="large")
def render_register_dialog():
    """渲染用于注册新人员的弹窗。"""
    st.subheader("新人员信息录入")
    with st.form("new_person_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("姓名", placeholder="例如：张三")
        sn = col2.text_input("唯一编号(SN)", placeholder="例如：EMP001")
        image_file = st.file_uploader("上传人脸照片", type=["jpg", "png", "jpeg"])

        if st.form_submit_button("✔️ 确认注册", type="primary", use_container_width=True):
            if name and sn and image_file:
                with st.spinner("正在注册新人员..."):
                    # ==================== 代码修复处 ====================
                    # 根据API文档，'name'和'sn'是query参数，应使用`params`传递
                    # 'image_file'是文件，应使用`files`传递

                    params_payload = {'name': name, 'sn': sn}
                    files_payload = {'image_file': (image_file.name, image_file.getvalue(), image_file.type)}

                    # 在api_request调用中，使用 `params` 而不是 `data`
                    success, data, msg = api_request(
                        'POST',
                        API_ENDPOINTS['FACES'],
                        params=params_payload,
                        files=files_payload
                    )
                    # =================================================

                if success:
                    st.toast(f"注册成功!", icon="🎉")
                    refresh_all_data()
                    st.session_state.show_register_dialog = False
                    st.rerun()
                else:
                    st.error(f"注册失败: {msg}")
            else:
                st.warning("所有字段均为必填项。")

    if st.button("取消", use_container_width=True):
        st.session_state.show_register_dialog = False
        st.rerun()


def render_dashboard_page():
    """渲染仪表盘页面。"""
    st.header("📊 仪表盘总览")
    faces_data = st.session_state.get("faces_data") or {}

    is_connected, _ = st.session_state.api_status
    if not is_connected:
        st.info("API服务未连接，请在左侧侧边栏配置正确的服务地址并确保后端服务已启动。")
        return  # 服务未连接时，不显示下面的内容

    if not faces_data.get("unique_sns"):
        st.info("人脸库为空，请先在“人脸库管理”页面注册新人员。")

    col1, col2, col3 = st.columns(3)
    with col1:
        unique_sns_count = len(faces_data.get('unique_sns', []))
        st.html(f"""
        <div class="info-card">
            <div class="icon">👥</div>
            <div class="title">人脸库人员总数</div>
            <div class="value">{unique_sns_count}</div>
        </div>
        """)
    with col2:
        api_status, api_color = ("在线", "#28a745") if st.session_state.api_status[0] else ("离线", "#dc3545")
        st.html(f"""
        <div class="info-card">
            <div class="icon">📡</div>
            <div class="title">API服务状态</div>
            <div class="value" style="color:{api_color};">{api_status}</div>
        </div>
        """)
    with col3:
        success, data, msg = api_request("GET", API_ENDPOINTS['STREAMS_LIST'])
        stream_count = data.get('active_streams_count', 0) if success else 0
        st.html(f"""
        <div class="info-card">
            <div class="icon">📹</div>
            <div class="title">当前活动视频流</div>
            <div class="value">{stream_count}</div>
        </div>
        """)

    st.divider()
    st.header("🧐 快速人脸识别")
    with st.container(border=True, height=450):
        uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "png", "jpeg"], key="recognize_uploader")
        if uploaded_file:
            col_img, col_res = st.columns([0.6, 0.4])
            with col_img:
                st.image(uploaded_file, caption="待识别图片预览", use_container_width=True)
            with col_res:
                st.subheader("识别结果")
                with st.spinner("正在识别中..."):
                    files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                    success, data, msg = api_request('POST', API_ENDPOINTS['RECOGNIZE'], files=files)
                    if success and data:
                        st.success(f"识别成功！找到 {len(data)} 个匹配项。")
                        for result in data:
                            with st.container(border=True):
                                st.markdown(f"**姓名:** {result.get('name')} | **SN:** {result.get('sn')}")
                                st.markdown(
                                    f"**相似度:** <span style='color:green;'>{1 - result.get('distance', 1):.2%}</span> (距离: {result.get('distance', 0):.4f})",
                                    unsafe_allow_html=True)
                    elif success and not data:
                        st.info("图像中可能未检测到人脸，或未在库中找到匹配项。")
                    else:
                        st.error(f"识别失败: {msg}")


def render_management_page():
    """渲染人脸库管理页面。"""
    st.header("🗂️ 人脸库管理中心")
    if st.button("➕ 注册新人员", type="primary"):
        st.session_state.show_register_dialog = True
        st.rerun()
    if st.session_state.get("show_register_dialog"):
        render_register_dialog()
    st.markdown("---")

    faces_data = st.session_state.get("faces_data") or {}
    if not faces_data.get('unique_sns'):
        st.info("人脸库为空，或数据加载中... 请确保API服务在线并尝试强制刷新数据。")
        return

    unique_sns = faces_data.get('unique_sns', [])
    all_faces_info = faces_data.get('faces', [])
    st.subheader(f"👥 人员列表 (共 {len(unique_sns)} 人)")

    # 每行显示的最佳列数
    num_cols = 3
    cols = st.columns(num_cols)
    for i, sn in enumerate(unique_sns):
        col = cols[i % num_cols]
        person_faces = [f for f in all_faces_info if f['sn'] == sn]
        if not person_faces: continue
        name = person_faces[0]['name']

        with col:
            with st.container(border=True, height=350):
                st.markdown(f"#### {name}")
                st.caption(f"SN: {sn}")
                st.metric(label="已注册人脸数", value=len(person_faces))

                # 最多预览3张图片
                if person_faces:
                    img_cols = st.columns(min(3, len(person_faces)))
                    for j, face_info in enumerate(person_faces[:3]):
                        with img_cols[j]:
                            img_url = convert_path_to_url(face_info.get('image_path'))
                            st.image(img_url, width=80, caption=f"ID: ...{face_info['uuid'][-4:]}")

                with st.expander("⚙️ 管理此人"):
                    with st.form(key=f"update_{sn}"):
                        new_name = st.text_input("新姓名", value=name, label_visibility="collapsed")
                        if st.form_submit_button("更新姓名", use_container_width=True, type="primary"):
                            if new_name and new_name != name:
                                with st.spinner("正在更新..."):
                                    endpoint = API_ENDPOINTS['FACE_BY_SN'].format(sn)
                                    success, data, msg = api_request('PUT', endpoint, json={"name": new_name})
                                if success:
                                    st.toast(f"'{name}' 已更新为 '{new_name}'", icon="✅")
                                    refresh_all_data()
                                    st.rerun()
                                else:
                                    st.error(f"更新失败: {msg}")

                    st.markdown("---")
                    confirm_delete = st.checkbox("我确认要删除此人所有记录", key=f"delete_confirm_{sn}")
                    if st.button("🗑️ 删除此人", key=f"delete_{sn}", use_container_width=True,
                                 disabled=not confirm_delete, type="secondary"):
                        with st.spinner("正在删除..."):
                            endpoint = API_ENDPOINTS['FACE_BY_SN'].format(sn)
                            success, _, msg = api_request('DELETE', endpoint)
                            if success:
                                st.toast(f"'{name}' ({sn}) 已被删除。", icon="🗑️")
                                refresh_all_data()
                                if st.session_state.viewing_stream_info and st.session_state.viewing_stream_info[
                                    'stream_id'] == sn:
                                    st.session_state.viewing_stream_info = None
                                st.rerun()
                            else:
                                st.error(f"删除失败: {msg}")


def render_monitoring_page():
    """【全新重构】渲染实时视频监控页面，支持多路流的查看与管理"""
    st.header("🛰️ 实时视频监测")

    # --- 1. 启动新监测任务的表单 ---
    with st.expander("▶️ 启动新监测任务", expanded=True):
        with st.form("start_stream_form"):
            col1, col2 = st.columns([2, 1])
            source = col1.text_input("视频源", "0", help="可以是摄像头ID(如 0, 1) 或 视频文件/URL")
            lifetime = col2.number_input("生命周期(分钟)", min_value=-1, value=10, help="-1 代表永久")

            if st.form_submit_button("🚀 开启监测", use_container_width=True, type="primary"):
                with st.spinner("正在请求启动视频流..."):
                    payload = {"source": source, "lifetime_minutes": lifetime}
                    success, data, msg = api_request('POST', API_ENDPOINTS['STREAMS_START'], json=payload)
                    if success and data:
                        st.toast(f"视频流任务已启动！ID: ...{data['stream_id'][-6:]}", icon="🚀")
                        st.session_state.viewing_stream_info = data
                        st.rerun()
                    else:
                        st.error(f"启动失败: {msg}")

    # --- 2. 显示当前正在观看的视频流 ---
    if st.session_state.get("viewing_stream_info"):
        stream_info = st.session_state.viewing_stream_info
        st.subheader(f"正在播放: {stream_info['source']}")
        st.caption(f"Stream ID: `{stream_info['stream_id']}`")
        # 直接使用后端返回的完整URL
        st.image(stream_info['feed_url'], caption=f"实时视频流 | 源: {stream_info['source']}")
    else:
        st.info("当前未选择任何视频流进行观看。请从下面的列表中选择一个，或启动一个新的监测任务。")

    st.divider()

    # --- 3. 【全新设计】获取并显示所有活动的视频流列表 ---
    st.subheader("所有活动中的监测任务")
    success, data, msg = api_request("GET", API_ENDPOINTS['STREAMS_LIST'])

    if not success:
        st.error(f"无法获取活动流列表: {msg}")
        return

    active_streams = data.get('streams', [])
    if not active_streams:
        st.info("目前没有正在运行的视频监测任务。")
    else:
        st.info(f"共发现 {len(active_streams)} 路活动视频流。")
        for stream in active_streams:
            stream_id = stream['stream_id']
            with st.container(border=True):
                # 使用多列布局优化显示
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])

                with col1:
                    st.markdown(f"**来源:** `{stream['source']}`")
                    st.caption(f"ID: `{stream_id[-12:]}`")

                with col2:
                    st.markdown(f"**启动于:** `{format_datetime_human(stream.get('started_at'))}`")

                with col3:
                    st.markdown(f"**将过期:** `{format_datetime_human(stream.get('expires_at'))}`")

                with col4:
                    # 将按钮放在同一行
                    b_col1, b_col2 = st.columns(2)

                    # 【核心修正】点击“观看”时，直接使用API返回的完整stream对象
                    if b_col1.button("👁️", key=f"view_{stream_id}", help="观看此流", use_container_width=True):
                        st.session_state.viewing_stream_info = stream
                        st.rerun()

                    if b_col2.button("⏹️", key=f"stop_{stream_id}", help="停止此流", type="primary",
                                     use_container_width=True):
                        with st.spinner(f"正在停止流 {stream['source']}..."):
                            endpoint = API_ENDPOINTS['STREAMS_STOP'].format(stream_id)
                            stop_success, _, stop_msg = api_request('POST', endpoint)
                            if stop_success:
                                st.toast(f"视频流 {stream['source']} 已停止。", icon="✅")
                                if st.session_state.viewing_stream_info and st.session_state.viewing_stream_info[
                                    'stream_id'] == stream_id:
                                    st.session_state.viewing_stream_info = None
                                st.rerun()
                            else:
                                st.error(f"停止失败: {stop_msg}")


# ==============================================================================
# 5. 主程序入口 (Main Application Entrypoint)
# ==============================================================================
def main():
    """主应用函数。"""
    initialize_session_state()
    render_sidebar()

    is_connected, _ = st.session_state.api_status
    if st.session_state.get("faces_data") is None and is_connected:
        refresh_all_data()

    st.title("欢迎来到星尘AI视觉平台")

    pages = ["仪表盘", "人脸库管理", "实时监测"]
    # 使用 st.radio 模拟的顶部导航栏
    st.session_state.active_page = st.radio(
        "主导航",
        options=pages,
        key="page_selector",
        label_visibility="collapsed",
        horizontal=True,
    )

    if st.session_state.active_page == "仪表盘":
        render_dashboard_page()
    elif st.session_state.active_page == "人脸库管理":
        render_management_page()
    elif st.session_state.active_page == "实时监测":
        render_monitoring_page()


if __name__ == "__main__":
    main()