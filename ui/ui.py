# 人脸识别智能管理系统 - 增强版UI
import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any, Dict, List, Optional
import os
import json
from datetime import datetime, date
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import time

# ==============================================================================
# 1. 页面配置与美化 (Page Config & Styling)
# ==============================================================================
st.set_page_config(
    page_title="人脸识别智能管理系统",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全新设计的CSS样式 ---
st.markdown("""
<style>
    /* --- 全局与字体 --- */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

    /* 统一浅色主题变量 */
    :root {
        --bg-primary: #f8f9fa;
        --bg-secondary: #e9ecef;
        --bg-card: #ffffff;
        --bg-sidebar: #ffffff;
        --text-primary: #1a1f36;
        --text-secondary: #6c757d;
        --border-color: #e0e4e8;
        --shadow-color: rgba(0,0,0,0.1);
        --accent-color: #4f46e5;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
        --info-color: #3b82f6;
        --light-accent: #e0e7ff;
        --light-success: #d1fae5;
        --light-error: #fee2e2;
        --light-warning: #fef3c7;
    }

    html, body, [class*="st-"] {
        font-family: 'Noto Sans SC', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
    }

    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        text-shadow: 0 1px 2px var(--shadow-color);
    }

    /* --- 侧边栏美化 --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg-secondary) 100%);
        border-right: 2px solid var(--border-color);
        box-shadow: 4px 0px 20px var(--shadow-color);
        color: var(--text-primary);
    }
    .st-emotion-cache-16txtl3 {
         padding-top: 2rem;
    }

    /* --- 自定义导航栏 --- */
    .nav-item button {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        text-align: left;
        padding: 12px 18px !important;
        margin-bottom: 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .nav-item button[kind="secondary"] {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--border-color) 100%);
        color: var(--text-secondary);
        border: 2px solid var(--border-color);
    }
    .nav-item button[kind="secondary"]:hover {
        background: linear-gradient(135deg, var(--border-color) 0%, var(--accent-color) 100%);
        border-color: var(--accent-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--accent-color);
        opacity: 0.8;
    }
    .nav-item button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-color) 0%, #4338ca 100%);
        color: white;
        border: 2px solid var(--accent-color);
        box-shadow: 0 4px 12px var(--accent-color);
    }
    .nav-item button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
        border-color: #4338ca;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--accent-color);
    }
    .nav-item button span {
        margin-right: 12px;
    }

    /* --- 指标卡片 (Metric Card) --- */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border-radius: 16px;
        padding: 30px;
        border: 2px solid var(--border-color);
        box-shadow: 0 8px 25px var(--shadow-color);
        transition: all 0.3s ease;
        height: 180px;
        min-height: 180px;
        position: relative;
        overflow: hidden;
        color: var(--text-primary);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), #06b6d4, var(--success-color));
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 40px var(--shadow-color);
        border-color: var(--accent-color);
    }
    .metric-card .title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 15px;
        text-align: center;
        flex-shrink: 0;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        text-align: center;
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px 0;
    }
    .metric-card.ok { border-left: 6px solid var(--success-color); }
    .metric-card.error { border-left: 6px solid var(--error-color); }
    .metric-card.info { border-left: 6px solid var(--info-color); }
    .metric-card.action { border-left: 6px solid var(--accent-color); }

    .metric-card .status {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: auto;
        font-weight: 500;
        text-align: center;
        flex-shrink: 0;
    }

    .metric-card.action {
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .metric-card.action:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px var(--shadow-color);
        border-color: var(--accent-color);
    }

    /* --- 通用按钮美化 --- */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .stButton>button:not([kind="secondary"]) {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        border: 2px solid #4f46e5;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .stButton>button:not([kind="secondary"]):hover {
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
        border-color: #4338ca;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
    }
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #495057;
        border: 2px solid #dee2e6;
    }
    .stButton>button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        border-color: #4f46e5;
        transform: translateY(-2px);
    }

    /* --- 容器和 Expander --- */
    .st-emotion-cache-1r6slb0 {
        border-radius: 16px;
        border: 2px solid #e0e4e8;
        padding: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 2px solid #e0e4e8 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* --- 标签页 (Tabs) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 3px solid #dee2e6;
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px 12px 0 0;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0 1rem;
        height: 60px;
        background: transparent;
        border-bottom: 4px solid transparent;
        transition: all 0.3s ease;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #4f46e5;
        color: #4f46e5;
        font-weight: 700;
        background: rgba(79, 70, 229, 0.1);
    }

    /* --- 错误和成功消息样式 --- */
    .success-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        color: #065f46;
        padding: 16px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    .error-message {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        color: #7f1d1d;
        padding: 16px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }

    /* 列布局优化 */
    [data-testid="column"] {
        padding: 0 10px;
    }

    /* 分页控件样式 */
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0;
    }

    .pagination-button {
        min-width: 45px;
        height: 45px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .pagination-button.active {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }

    /* 相似度颜色提示 */
    .similarity-badge {
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .similarity-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .similarity-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }

    .similarity-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    /* 数据表格美化 */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* 文件上传美化 */
    .stFileUploader {
        border-radius: 12px;
        border: 2px dashed var(--accent-color);
        background: rgba(79, 70, 229, 0.05);
        padding: 20px;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #4338ca;
        background: rgba(79, 70, 229, 0.1);
    }

    /* 表格美化 - 支持深色模式 */
    .stDataFrame {
        background-color: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    /* 表单美化 - 支持深色模式 */
    .stTextInput, .stSelectbox, .stDateInput {
        background-color: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    /* 容器美化 - 支持深色模式 */
    .stContainer {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }

</style>

<script>
// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    // 可以在这里添加其他页面初始化逻辑
    console.log('人脸识别系统UI已加载完成');
});
</script>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. 增强的API客户端 (Enhanced API Client)
# ==============================================================================
class ApiClient:
    """增强的API客户端类，提供完整的错误处理和数据验证"""

    def __init__(self, base_url: str):
        self.base_url = f"http://{base_url.replace('http://', '')}"
        self.endpoints = {
            'health': '/api/face/health',
            'faces': '/api/face/faces',
            'face_by_sn': '/api/face/faces/{}',
            'recognize': '/api/face/recognize',
            'streams_start': '/api/face/streams/start',
            'streams_stop': '/api/face/streams/stop/{}',  # 使用 task_id
            'streams_list': '/api/face/streams',
            'records': '/api/detection/records',
            'stats': '/api/detection/stats',
            'weekly_trend': '/api/detection/weekly-trend',
            'person_pie': '/api/detection/person-pie',
            'hourly_trend': '/api/detection/hourly-trend',
            'top_persons': '/api/detection/top-persons',
        }

    def _request(self, method: str, endpoint_key: str, **kwargs) -> Tuple[bool, Any]:
        """统一的内部请求方法，增强错误处理"""
        url = f"{self.base_url}{kwargs.pop('url_format', self.endpoints[endpoint_key])}"

        try:
            response = requests.request(method, url, timeout=30, **kwargs)

            if response.ok:
                if response.status_code == 204 or not response.content:
                    return True, {"msg": "操作成功"}

                try:
                    res_json = response.json()
                    if res_json.get("code") == 0:
                        return True, res_json.get("data", {})
                    else:
                        return False, res_json.get("msg", "后端返回业务错误")
                except json.JSONDecodeError:
                    return False, "服务器返回了无效的JSON响应"
            else:
                try:
                    error_detail = response.json()
                    if isinstance(error_detail.get("detail"), list):
                        detail = error_detail["detail"][0].get('msg', '请求验证失败')
                    else:
                        detail = error_detail.get("detail", "未知错误")
                    return False, f"HTTP {response.status_code}: {detail}"
                except json.JSONDecodeError:
                    return False, f"HTTP {response.status_code}: 服务器错误"

        except requests.exceptions.Timeout:
            return False, "请求超时，请检查网络连接或服务器状态"
        except requests.exceptions.ConnectionError:
            return False, "无法连接到服务器，请检查服务器地址和状态"
        except requests.RequestException as e:
            return False, f"网络请求失败: {str(e)}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"

    # 基础接口方法
    def check_health(self):
        return self._request('GET', 'health')

    def get_all_faces(self):
        return self._request('GET', 'faces')

    def register_face(self, data, files):
        return self._request('POST', 'faces', data=data, files=files)

    def update_face(self, sn, name):
        return self._request('PUT', 'face_by_sn',
                             url_format=self.endpoints['face_by_sn'].format(sn),
                             json={"name": name})

    def delete_face(self, sn):
        return self._request('DELETE', 'face_by_sn',
                             url_format=self.endpoints['face_by_sn'].format(sn))

    def recognize_face(self, files):
        return self._request('POST', 'recognize', files=files)

    def start_stream(self, payload: dict):
        """启动视频流，使用完整的payload"""
        return self._request('POST', 'streams_start', json=payload)

    def stop_stream(self, task_id: int):
        """根据task_id停止视频流"""
        return self._request('POST', 'streams_stop',
                             url_format=self.endpoints['streams_stop'].format(task_id))

    def list_streams(self):
        return self._request('GET', 'streams_list')

    # 检测统计接口方法 - 增强版本
    def get_detection_stats(self):
        """获取检测统计信息"""
        return self._request('GET', 'stats')

    def get_weekly_trend(self):
        """获取周趋势数据"""
        return self._request('GET', 'weekly_trend')

    def get_detection_records(self, params):
        """获取检测记录列表"""
        # 确保参数格式正确
        clean_params = {k: v for k, v in params.items() if v is not None}
        return self._request('GET', 'records', params=clean_params)

    def get_person_pie_data(self):
        """获取人员检测饼图数据"""
        return self._request('GET', 'person_pie')

    def get_hourly_trend_data(self):
        """获取小时趋势数据"""
        return self._request('GET', 'hourly_trend')

    def get_top_persons_data(self, limit=10):
        """获取检测排行数据"""
        return self._request('GET', 'top_persons', params={'limit': limit})


# ==============================================================================
# 3. 会话状态管理 (Session State)
# ==============================================================================
def initialize_session_state():
    """初始化应用所需的全部会话状态"""
    if "app_state" not in st.session_state:
        backend_host = os.getenv("HOST__IP", "localhost")
        backend_port = os.getenv("SERVER__PORT", "12010")
        st.session_state.app_state = {
            "api_url": f"{backend_host}:{backend_port}",
            "api_client": ApiClient(f"{backend_host}:{backend_port}"),
            "api_status": (False, "尚未连接"),
            "active_page": "数据看板",
            "management": {
                "search_query": "",
                "selected_sn": None,
            },
            "monitoring": {
                "viewing_stream_info": None
            },
            "analytics": {
                "records_page": 1
            }
        }

    # 初始化主题设置
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    # 初始化页面切换状态
    if "page_changed" not in st.session_state:
        st.session_state.page_changed = False

    # 初始化数据刷新状态
    if "data_refresh_trigger" not in st.session_state:
        st.session_state.data_refresh_trigger = False


# ==============================================================================
# 4. 工具函数 (Utility Functions)
# ==============================================================================
def safe_format_datetime(dt_str):
    """安全地格式化日期时间字符串"""
    try:
        if isinstance(dt_str, str):
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return str(dt_str)
    except:
        return str(dt_str)


def display_error_message(message: str):
    """显示错误消息"""
    st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)


def display_success_message(message: str):
    """显示成功消息"""
    st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)


def create_empty_chart(chart_type="bar", title="暂无数据"):
    """创建空图表占位符"""
    if chart_type == "bar":
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[], y=[]))
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            height=350,
            showlegend=False
        )
        return fig
    elif chart_type == "pie":
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=[], values=[]))
        fig.update_layout(
            title=title,
            height=350,
            showlegend=False
        )
        return fig
    return None


def get_similarity_color(similarity: float) -> str:
    """根据相似度值返回对应的渐变色（红到绿）"""
    if similarity < 0 or similarity > 1:
        return "#f0f0f0"  # 无效值的默认颜色

    # 计算RGB颜色（红到绿渐变）
    r = int(255 * (1 - similarity))
    g = int(255 * similarity)
    b = 0

    # 确保颜色值在有效范围内
    r = max(0, min(255, r))
    g = max(0, min(255, g))

    return f"rgb({r}, {g}, {b})"


def format_similarity_display(similarity: float) -> str:
    """格式化相似度显示（带颜色标记）"""
    color = get_similarity_color(similarity)
    return f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{similarity:.2%}</span>'


# ==============================================================================
# 5. UI 渲染模块 (全面修复和美化)
# ==============================================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("✨ 人脸识别系统")
        st.caption("v0.1.0")

        # API连接配置
        st.markdown("### 🔗 API连接配置")
        api_url = st.text_input("后端服务地址", value=st.session_state.app_state['api_url'],
                                placeholder="localhost:12010", key="api_url_input")

        # 使用on_change来避免不必要的rerun
        if api_url != st.session_state.app_state['api_url']:
            st.session_state.app_state['api_url'] = api_url
            st.session_state.app_state['api_client'] = ApiClient(api_url)
            # 清除相关缓存，但不rerun
            st.cache_data.clear()
            st.toast("API地址已更新", icon="✅")

        # 健康检查
        client = st.session_state.app_state['api_client']
        success, data = client.check_health()

        if success:
            status_msg = data.get('message', "连接成功") if isinstance(data, dict) else "连接成功"
            status_icon = "✅"
            status_color = "success"
        else:
            status_msg = str(data)
            status_icon = "❌"
            status_color = "error"

        st.session_state.app_state['api_status'] = (success, status_msg)

        # 美化状态显示
        if success:
            st.success(f"**API状态:** {status_msg}", icon=status_icon)
        else:
            st.error(f"**API状态:** {status_msg}", icon=status_icon)

        st.divider()

        # 导航菜单
        st.markdown("### 🧭 系统导航")
        pages = {
            "数据看板": "📊",
            "人脸库管理": "🗂️",
            "实时监测": "🛰️",
            "统计图表": "📈",
            "历史记录": "📋"
        }

        for page, icon in pages.items():
            st.markdown(f'<div class="nav-item">', unsafe_allow_html=True)
            if st.button(
                    f"{icon} {page}",
                    use_container_width=True,
                    type="primary" if st.session_state.app_state['active_page'] == page else "secondary",
                    key=f"nav_{page}"  # 添加唯一key
            ):
                st.session_state.app_state['active_page'] = page
                # 使用session_state来管理页面切换，避免rerun
                st.session_state.page_changed = True
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # 系统工具
        st.markdown("### 🛠️ 系统工具")

        # 系统状态指示器
        st.markdown("### 📊 系统状态")
        col1, col2 = st.columns(2)

        with col1:
            # 显示当前API状态
            api_status = st.session_state.app_state['api_status'][0]
            if api_status:
                st.success("🟢 API连接正常", icon="✅")
            else:
                st.error("🔴 API连接异常", icon="❌")

        with col2:
            # 显示缓存状态
            cache_status = "正常" if not st.session_state.data_refresh_trigger else "刷新中"
            st.info(f"💾 缓存状态: {cache_status}", icon="💾")

        if st.button("🔄 强制刷新全站数据", use_container_width=True, type="secondary", key="refresh_data"):
            st.cache_data.clear()
            st.toast("数据缓存已清除，正在刷新...", icon="🔄")
            # 使用session_state来触发刷新，避免rerun
            st.session_state.data_refresh_trigger = True

        if st.button("📊 查看API文档", use_container_width=True, type="secondary"):
            st.info("API文档地址: /docs")

        st.divider()
        st.caption("© 2024 人脸识别智能管理系统")


def render_dashboard_page():
    """渲染数据看板页面 - 深度优化版本"""
    # 页面标题和快速操作
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("📊 智能数据看板")
        st.markdown("实时监控系统状态，快速了解关键指标")

    with col2:
        if st.button("🔄 刷新数据", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.toast("数据已刷新", icon="✅")

    st.markdown("---")

    client = st.session_state.app_state['api_client']

    @st.cache_data(ttl=30)
    def get_dashboard_data():
        """获取看板数据"""
        try:
            # 获取检测统计信息
            stats_s, stats_d = client.get_detection_stats()
            if not stats_s:
                st.warning(f"⚠️ 检测统计获取失败: {stats_d}")

            # 获取人脸库信息
            faces_s, faces_d = client.get_all_faces()
            if not faces_s:
                st.warning(f"⚠️ 人脸库信息获取失败: {faces_d}")

            # 获取趋势数据
            trend_s, trend_d = client.get_weekly_trend()
            if not trend_s:
                st.warning(f"⚠️ 趋势数据获取失败: {trend_d}")

            # 调试信息
            if st.session_state.app_state.get('api_url', '').startswith('localhost'):
                with st.expander("🔍 调试信息（开发模式）", expanded=False):
                    st.write("检测统计:", "成功" if stats_s else f"失败: {stats_d}")
                    st.write("人脸库:", "成功" if faces_s else f"失败: {faces_d}")
                    st.write("趋势数据:", "成功" if trend_s else f"失败: {trend_d}")
                    if stats_s and stats_d:
                        st.json(stats_d)
                    if faces_s and faces_d:
                        st.json(faces_d)
                    if trend_s and trend_d:
                        st.json(trend_d)

            return {
                "stats": stats_d if stats_s else {},
                "faces": faces_d if faces_s else {},
                "trend": trend_d if trend_s else {},
                "errors": {
                    "stats": None if stats_s else stats_d,
                    "faces": None if faces_s else faces_d,
                    "trend": None if trend_s else trend_d
                }
            }
        except Exception as e:
            st.error(f"❌ 获取看板数据时发生错误: {str(e)}")
            return {"stats": {}, "faces": {}, "trend": {}, "errors": {}}

    data = get_dashboard_data()

    # 显示错误信息（如果有）
    if any(data.get("errors", {}).values()):
        with st.expander("⚠️ 数据加载警告", expanded=False):
            for key, error in data.get("errors", {}).items():
                if error:
                    st.warning(f"{key} 数据加载失败: {error}")

    # API状态
    api_status, api_color_class = ("在线", "ok") if st.session_state.app_state['api_status'][0] else ("离线", "error")

    # 核心指标卡片 - 深度优化版本
    st.markdown("### 🎯 核心指标概览")

    # 获取统计数据，提供默认值
    stats = data.get('stats', {})
    faces = data.get('faces', {})

    # 核心指标 - 只保留4个主要指标，确保完美对齐
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        # 人脸库人员总数
        if faces and 'faces' in faces and faces['faces']:
            unique_persons = len(set(face.get('sn') for face in faces['faces'] if face.get('sn')))
            count = unique_persons
        else:
            count = stats.get('unique_persons', 0)

        if count == 0:
            count = "0"
            card_class = "error"
            status_text = "⚠️ 暂无人员"
        else:
            card_class = "ok"
            status_text = "✅ 正常"

        st.html(f"""
        <div class="metric-card {card_class}">
            <div class="title">👥 人脸库人员总数</div>
            <div class="value">{count}</div>
            <div class="status">{status_text}</div>
        </div>
        """)

    with col2:
        total = stats.get('total_detections', 0)
        if total == 0:
            total = "0"
            card_class = "error"
            status_text = "⚠️ 暂无检测"
        else:
            card_class = "ok"
            status_text = "✅ 正常"

        st.html(f"""
        <div class="metric-card {card_class}">
            <div class="title">🔍 总检测次数</div>
            <div class="value">{total}</div>
            <div class="status">{status_text}</div>
        </div>
        """)

    with col3:
        today = stats.get('today_detections', 0)
        if today == 0:
            today = "0"
            card_class = "error"
            status_text = "⚠️ 今日无检测"
        else:
            card_class = "ok"
            status_text = "✅ 今日活跃"

        st.html(f"""
        <div class="metric-card {card_class}">
            <div class="title">📅 今日检测</div>
            <div class="value">{today}</div>
            <div class="status">{status_text}</div>
        </div>
        """)

    with col4:
        st.html(f"""
        <div class="metric-card {api_color_class}">
            <div class="title">🌐 API 服务状态</div>
            <div class="value">{api_status}</div>
            <div class="status">{'✅ 连接正常' if api_status == '在线' else '❌ 连接异常'}</div>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # 趋势图表和最新记录
    st.markdown("### 📊 数据趋势分析")
    col1, col2 = st.columns([0.65, 0.35])

    with col1, st.container(border=True):
        st.subheader("🗓️ 近7日检测趋势")
        trend_data = data.get('trend', {}).get('trend_data', [])

        if trend_data:
            try:
                trend_df = pd.DataFrame(trend_data)
                trend_df['date'] = pd.to_datetime(trend_df['date'])

                # 使用Plotly创建更美观的趋势图
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df['date'],
                    y=trend_df['count'],
                    mode='lines+markers',
                    line=dict(color='#4f46e5', width=4),
                    marker=dict(size=8, color='#4f46e5'),
                    fill='tonexty',
                    fillcolor='rgba(79, 70, 229, 0.1)',
                    name='检测次数'
                ))

                fig.update_layout(
                    title="近7日检测趋势分析",
                    xaxis_title="日期",
                    yaxis_title="检测次数",
                    height=350,
                    showlegend=False,
                    plot_bgcolor='rgba(255, 255, 255, 0.9)',
                    paper_bgcolor='rgba(255, 255, 255, 0.9)',
                    font=dict(size=12, color='#1a1f36'),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"趋势图表渲染失败: {str(e)}")
                st.info("暂无趋势数据或数据格式错误。")
        else:
            st.info("暂无趋势数据。")

    with col2, st.container(border=True, height=430):
        st.subheader("⏱️ 最新检测记录")
        recent = data.get('stats', {}).get('recent_detections', [])

        if recent:
            try:
                for i, item in enumerate(recent[:5]):  # 只显示前5条
                    with st.container():
                        col_img, col_info = st.columns([0.3, 0.7])

                        # 安全地显示图片
                        try:
                            if item.get('image_url'):
                                col_img.image(item['image_url'], width=60, caption=f"记录 {i + 1}")
                            else:
                                col_img.write("📷 无图片")
                        except:
                            col_img.write("❌ 图片加载失败")

                        # 显示信息
                        name = item.get('name', 'Unknown')
                        col_info.markdown(f"**{name}**")

                        # 安全地格式化时间
                        time_str = safe_format_datetime(item.get('create_time', ''))
                        col_info.caption(f"🕐 {time_str}")

                        if i < len(recent) - 1:  # 不在最后一条记录后添加分隔线
                            st.markdown("---")
            except Exception as e:
                st.error(f"最新记录显示失败: {str(e)}")
        else:
            st.info("暂无最近检测记录。")

    st.markdown("---")

    # 快速人脸识别 - 美化版本
    st.markdown("### 🧐 快速人脸识别")
    with st.container(border=True):
        uploaded_file = st.file_uploader("📁 上传图片进行识别", type=["jpg", "png", "jpeg"],
                                         help="支持JPG、PNG、JPEG格式的图片文件")

        if uploaded_file:
            img_col, res_col = st.columns(2)

            # 显示上传的图片
            with img_col:
                st.markdown("**📸 待识别图片**")
                try:
                    st.image(uploaded_file, caption="上传的图片", width=300)
                except Exception as e:
                    st.error(f"❌ 图片显示失败: {str(e)}")

            # 识别按钮和结果显示
            with res_col:
                st.markdown("**🔍 识别结果**")
                if st.button("🚀 开始识别", type="primary", use_container_width=True,
                             help="点击开始人脸识别"):
                    with st.spinner("🔄 正在识别中，请稍候..."):
                        try:
                            files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                            success, results = client.recognize_face(files)

                            if success:
                                if isinstance(results, list) and len(results) > 0:
                                    st.success(f"🎉 识别成功！找到 {len(results)} 个匹配项")

                                    for i, res in enumerate(results, 1):
                                        with st.container(border=True):
                                            st.markdown(f"**🏆 匹配结果 {i}**")

                                            # 创建结果展示卡片
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown(f"**👤 姓名:** {res.get('name', 'Unknown')}")
                                                st.markdown(f"**🆔 SN:** {res.get('sn', 'Unknown')}")
                                            with col2:
                                                similarity = res.get('similarity', 0)
                                                confidence = res.get('detection_confidence', 0)

                                                # 相似度颜色显示
                                                if similarity >= 0.7:
                                                    sim_color = "🟢"
                                                elif similarity >= 0.4:
                                                    sim_color = "🟡"
                                                else:
                                                    sim_color = "🔴"

                                                st.markdown(f"**🎯 相似度:** {sim_color} {similarity:.2%}")
                                                st.markdown(f"**✅ 检测置信度:** {confidence:.2%}")
                                else:
                                    st.warning("🤔 检测到人脸，但未在库中找到匹配项")
                            else:
                                display_error_message(f"❌ 识别失败: {results}")

                        except Exception as e:
                            display_error_message(f"❌ 识别过程中发生错误: {str(e)}")


def render_management_page():
    """渲染人脸库管理页面"""
    st.header("🗂️ 人脸库管理中心")
    st.markdown("---")
    client = st.session_state.app_state['api_client']

    @st.dialog("➕ 注册新人员", width="large")
    def register_dialog():
        with st.form("register_form"):
            name = st.text_input("姓名", placeholder="例如：张三")
            sn = st.text_input("唯一编号(SN)", placeholder="例如：EMP001")
            image_file = st.file_uploader("上传人脸照片", type=["jpg", "png", "jpeg"])

            if st.form_submit_button("✔️ 确认注册", type="primary", use_container_width=True):
                if not all([name, sn, image_file]):
                    st.warning("所有字段均为必填项。")
                    return

                with st.spinner("注册中..."):
                    try:
                        success, msg = client.register_face(
                            data={'name': name, 'sn': sn},
                            files={'image_file': (image_file.name, image_file.getvalue())}
                        )

                        if success:
                            st.toast("注册成功！", icon="🎉")
                            st.cache_data.clear()
                            st.session_state.app_state['management']['selected_sn'] = None
                            st.rerun()
                        else:
                            display_error_message(f"注册失败: {msg}")
                    except Exception as e:
                        display_error_message(f"注册过程中发生错误: {str(e)}")

    if st.button("➕ 注册新人员"):
        register_dialog()

    st.divider()

    @st.cache_data(ttl=60)
    def get_faces_data():
        try:
            success, data = client.get_all_faces()
            if success and data.get('faces'):
                return pd.DataFrame(data['faces'])
            return pd.DataFrame()
        except Exception as e:
            st.error(f"获取人脸数据失败: {str(e)}")
            return pd.DataFrame()

    faces_df = get_faces_data()

    if faces_df.empty:
        st.info("人脸库为空或加载失败。")
        return

    # 按人员分组
    try:
        persons_df = faces_df.groupby('sn').agg(
            name=('name', 'first'),
            registrations=('uuid', 'count')
        ).reset_index()
    except Exception as e:
        st.error(f"数据处理失败: {str(e)}")
        return

    col_table, col_detail = st.columns([0.5, 0.5])

    with col_table:
        st.subheader(f"👥 人员列表 (共 {len(persons_df)} 人)")
        search_query = st.text_input("🔍 搜索姓名或SN", key="person_search", placeholder="输入关键词进行搜索...")

        if search_query:
            filtered_df = persons_df[
                persons_df['name'].str.contains(search_query, case=False, na=False) |
                persons_df['sn'].str.contains(search_query, case=False, na=False)
                ]
        else:
            filtered_df = persons_df

        if filtered_df.empty:
            st.info("未找到匹配的人员。")
        else:
            selection = st.dataframe(
                filtered_df,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                column_config={
                    "sn": "唯一编号 (SN)",
                    "name": "姓名",
                    "registrations": "照片数"
                },
                use_container_width=True,
                key="person_selector"
            )

            if selection.selection.rows:
                selected_index = selection.selection.rows[0]
                st.session_state.app_state['management']['selected_sn'] = filtered_df.iloc[selected_index]['sn']

    with col_detail, st.container(border=True, height=550):
        sn = st.session_state.app_state['management'].get('selected_sn')
        if sn:
            try:
                person_details = faces_df[faces_df['sn'] == sn]
                if not person_details.empty:
                    name = person_details.iloc[0]['name']

                    st.subheader(f"👤 {name}")
                    st.caption(f"SN: {sn}")
                    st.markdown("**已注册照片:**")

                    # 安全地显示图片
                    try:
                        image_urls = [row['image_url'] for _, row in person_details.iterrows() if row.get('image_url')]
                        if image_urls:
                            st.image(image_urls, width=80)
                        else:
                            st.info("暂无图片")
                    except Exception as e:
                        st.warning(f"图片加载失败: {str(e)}")

                    st.divider()
                    with st.expander("⚙️ 管理选项", expanded=True):
                        new_name = st.text_input("更新姓名", value=name, key=f"update_{sn}")

                        if st.button("✔️ 确认更新", key=f"update_btn_{sn}", use_container_width=True):
                            if new_name and new_name != name:
                                with st.spinner("更新中..."):
                                    try:
                                        success, msg = client.update_face(sn, new_name)
                                        if success:
                                            st.toast("更新成功", icon="✅")
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            display_error_message(f"更新失败: {msg}")
                                    except Exception as e:
                                        display_error_message(f"更新过程中发生错误: {str(e)}")

                        if st.button("🗑️ 删除此人所有记录", use_container_width=True, key=f"delete_{sn}",
                                     type="secondary", help="此操作将删除该人员的所有人脸记录"):
                            with st.spinner("删除中..."):
                                try:
                                    success, msg = client.delete_face(sn)
                                    if success:
                                        st.toast("删除成功", icon="🗑️")
                                        st.cache_data.clear()
                                        st.session_state.app_state['management']['selected_sn'] = None
                                        st.rerun()
                                    else:
                                        display_error_message(f"删除失败: {msg}")
                                except Exception as e:
                                    display_error_message(f"删除过程中发生错误: {str(e)}")
                else:
                    st.warning("选中的人员数据不存在")
            except Exception as e:
                display_error_message(f"加载人员详情失败: {str(e)}")
        else:
            st.info("请从左侧列表中选择一位人员以查看详细信息和管理选项。")


def render_monitoring_page():
    """渲染实时监测页面 - [已适配新接口]"""
    st.header("🛰️ 实时视频监测")
    st.markdown("---")
    client = st.session_state.app_state['api_client']

    with st.expander("▶️ 启动新监测任务", expanded=True):
        with st.form("start_stream_form"):
            st.write("**基础配置**")
            source = st.text_input("视频源", "0", help="摄像头ID(0, 1) 或 视频文件/URL")
            lifetime = st.number_input("生命周期(分钟)", min_value=-1, value=10, help="-1 代表永久")

            st.write("**任务参数**")
            # 生成一个基于时间的随机任务ID作为默认值
            default_task_id = int(time.time() * 1000) + random.randint(0, 999)

            col1, col2 = st.columns(2)
            task_id = col1.number_input("任务ID (TaskId)", min_value=1, value=default_task_id)
            app_id = col2.number_input("应用ID (AppId)", min_value=1, value=31)
            app_name = col1.text_input("应用名称 (AppName)", "人脸应用")
            domain_name = col2.text_input("域名 (DomainName)", "video.com")

            if st.form_submit_button("🚀 开启监测", use_container_width=True, type="primary"):
                payload = {
                    "source": source,
                    "lifetime_minutes": lifetime,
                    "taskId": task_id,
                    "appId": app_id,
                    "appName": app_name,
                    "domainName": domain_name
                }
                with st.spinner("请求启动视频流..."):
                    try:
                        success, data = client.start_stream(payload)
                        if success:
                            st.toast("视频流任务已启动！", icon="🚀")
                            st.session_state.app_state['monitoring']['viewing_stream_info'] = data
                            st.rerun()
                        else:
                            display_error_message(f"启动失败: {data}")
                    except Exception as e:
                        display_error_message(f"启动过程中发生错误: {str(e)}")

    # 当前观看的流
    stream_info = st.session_state.app_state['monitoring'].get('viewing_stream_info')
    if stream_info:
        with st.container(border=True):
            st.subheader(f"正在播放: `{stream_info.get('source', 'N/A')}`")
            st.caption(f"Task ID: `{stream_info.get('task_id', 'N/A')}`")
            try:
                st.image(stream_info['feed_url'])
            except Exception as e:
                st.error(f"视频流加载失败: {str(e)}")
    else:
        st.info("请从下方列表选择一个流进行观看，或启动一个新任务。")

    st.divider()

    # 活动流列表 - 修复版本
    st.subheader("所有活动中的监测任务")

    # 清除缓存以确保获取最新状态
    if 'stream_stop_clicked' in st.session_state:
        st.cache_data.clear()
        del st.session_state.stream_stop_clicked

    @st.cache_data(ttl=5)  # 缓存5秒避免频繁请求
    def get_active_streams():
        try:
            success, data = client.list_streams()
            if success and isinstance(data, dict):
                return data.get('streams', [])
            return []
        except Exception as e:
            st.error(f"获取活动流失败: {str(e)}")
            return []

    active_streams = get_active_streams()

    if not active_streams:
        st.info("目前没有正在运行的视频监测任务。")
    else:
        for i, stream in enumerate(active_streams):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    try:
                        # 处理流信息显示
                        source = stream.get('source', '未知来源')
                        task_id = stream.get('task_id', f'unknown_{i}')  # 使用 task_id

                        # 处理过期时间
                        expires_at = None
                        if stream.get('expires_at'):
                            try:
                                expires_at = datetime.fromisoformat(stream['expires_at'].replace('Z', '+00:00'))
                            except:
                                expires_at = None

                        expires_display = expires_at.strftime('%Y-%m-%d %H:%M:%S') if expires_at else "永久"

                        st.markdown(f"**来源:** `{source}` | **过期时间:** {expires_display}")
                        st.caption(f"Task ID: `{task_id}`")  # 使用 task_id

                        # 显示流状态
                        status = "运行中"
                        status_color = "green"
                        st.markdown(f"**状态:** :{status_color}[{status}]")

                    except Exception as e:
                        st.warning(f"流信息显示异常: {str(e)}")

                with col2:
                    btn_cols = st.columns(2)

                    # 观看按钮
                    if btn_cols[0].button("👁️", key=f"view_{task_id}_{i}",  # 使用 task_id
                                          help="观看此流", use_container_width=True):
                        st.session_state.app_state['monitoring']['viewing_stream_info'] = stream
                        st.rerun()

                    # 停止按钮 - 修复停止功能
                    stop_key = f"stop_{task_id}_{i}"  # 使用 task_id
                    if btn_cols[1].button("⏹️", key=stop_key,
                                          help="停止此流", type="secondary", use_container_width=True):
                        st.session_state.stream_stop_clicked = True
                        with st.spinner("停止中..."):
                            try:
                                success, result = client.stop_stream(task_id)  # 使用 task_id
                                if success:
                                    st.toast(f"视频流 {task_id} 已停止。", icon="✅")
                                    # 清除当前观看的流如果是同一个
                                    current_stream = st.session_state.app_state['monitoring'].get('viewing_stream_info')
                                    if current_stream and current_stream.get('task_id') == task_id:  # 使用 task_id
                                        st.session_state.app_state['monitoring']['viewing_stream_info'] = None
                                    # 强制刷新页面
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    display_error_message(f"停止失败: {result}")
                            except Exception as e:
                                display_error_message(f"停止过程中发生错误: {str(e)}")


def render_analytics_page():
    """渲染检测分析页面 - 全面修复版本"""
    st.header("🔍 检测分析中心")
    st.markdown("---")
    client = st.session_state.app_state['api_client']

    # 数据获取函数 - 增强错误处理
    @st.cache_data(ttl=30)
    def get_analytics_data():
        """获取分析数据"""
        try:
            pie_s, pie_d = client.get_person_pie_data()
            top_s, top_d = client.get_top_persons_data(limit=10)
            hourly_s, hourly_d = client.get_hourly_trend_data()

            return {
                "pie": (pie_s, pie_d),
                "top": (top_s, top_d),
                "hourly": (hourly_s, hourly_d),
            }
        except Exception as e:
            st.error(f"获取分析数据时发生错误: {str(e)}")
            return {
                "pie": (False, "数据获取失败"),
                "top": (False, "数据获取失败"),
                "hourly": (False, "数据获取失败"),
            }

    analytics_data = get_analytics_data()

    tab1, tab2 = st.tabs(["📊 统计图表", "🗂️ 历史记录"])

    with tab1:
        col1, col2 = st.columns(2)

        # 人员检测分布饼图 - 修复版本
        with col1, st.container(border=True, height=450):
            st.subheader("👥 人员检测分布")
            success, data = analytics_data["pie"]

            if not success:
                display_error_message(f"加载饼图数据失败: {data}")
                st.plotly_chart(create_empty_chart("pie", "暂无人员分布数据"), use_container_width=True)
            elif data and data.get('pie_data'):
                try:
                    df = pd.DataFrame(data['pie_data'])

                    # 处理小比例数据
                    df.loc[df['percentage'] < 2, 'name'] = '其他'
                    df = df.groupby('name')['count'].sum().reset_index()

                    # 使用Plotly创建饼图
                    fig = px.pie(df, values='count', names='name',
                                 title="人员检测分布",
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                        height=350,
                        showlegend=True,
                        plot_bgcolor='rgba(255, 255, 255, 0.9)',
                        paper_bgcolor='rgba(255, 255, 255, 0.9)',
                        font=dict(color='#1a1f36')
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    display_error_message(f"饼图渲染失败: {str(e)}")
                    st.plotly_chart(create_empty_chart("pie", "图表渲染失败"), use_container_width=True)
            else:
                st.info("暂无人员检测分布数据。")
                st.plotly_chart(create_empty_chart("pie", "暂无人员分布数据"), use_container_width=True)

        # 检测次数排行榜 - 修复版本
        with col2, st.container(border=True, height=450):
            st.subheader("🏆 检测次数排行榜 (Top 10)")
            success, data = analytics_data["top"]

            if not success:
                display_error_message(f"加载排行榜数据失败: {data}")
                st.plotly_chart(create_empty_chart("bar", "暂无排行数据"), use_container_width=True)
            elif data and data.get('top_persons'):
                try:
                    df = pd.DataFrame(data['top_persons'])

                    # 使用Plotly创建水平条形图
                    fig = px.bar(df, x='count', y='name', orientation='h',
                                 title="检测次数排行榜",
                                 labels={'count': '检测次数', 'name': '姓名'},
                                 color='count',
                                 color_continuous_scale='Blues')

                    fig.update_layout(
                        height=350,
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False,
                        plot_bgcolor='rgba(255, 255, 255, 0.9)',
                        paper_bgcolor='rgba(255, 255, 255, 0.9)',
                        font=dict(color='#1a1f36')
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{y}</b><br>检测次数: %{x}<extra></extra>'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    display_error_message(f"排行榜渲染失败: {str(e)}")
                    st.plotly_chart(create_empty_chart("bar", "图表渲染失败"), use_container_width=True)
            else:
                st.info("暂无排行数据。")
                st.plotly_chart(create_empty_chart("bar", "暂无排行数据"), use_container_width=True)

        # 24小时检测活跃度 - 修复版本
        with st.container(border=True):
            st.subheader("🕒 24小时检测活跃度")
            success, data = analytics_data["hourly"]

            if not success:
                display_error_message(f"加载小时趋势数据失败: {data}")
                st.plotly_chart(create_empty_chart("bar", "暂无小时趋势数据"), use_container_width=True)
            elif data and data.get('hourly_data'):
                try:
                    df = pd.DataFrame(data['hourly_data'])

                    # 使用Plotly创建柱状图
                    fig = px.bar(df, x='hour', y='count',
                                 title="24小时检测活跃度",
                                 labels={'hour': '小时', 'count': '检测次数'},
                                 color='count',
                                 color_continuous_scale='Blues')

                    fig.update_layout(
                        height=300,
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                        showlegend=False
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{x}:00</b><br>检测次数: %{y}<extra></extra>'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    display_error_message(f"小时趋势图渲染失败: {str(e)}")
                    st.plotly_chart(create_empty_chart("bar", "图表渲染失败"), use_container_width=True)
            else:
                st.info("暂无小时趋势数据。")
                st.plotly_chart(create_empty_chart("bar", "暂无小时趋势数据"), use_container_width=True)

    # 历史记录查询 - 修复版本（解决跳转问题）
    with tab2:
        st.subheader("历史检测记录查询")

        # 初始化会话状态
        if 'analytics_filter_params' not in st.session_state:
            st.session_state.analytics_filter_params = {
                "name": "",
                "sn": "",
                "start_date": None,
                "end_date": None
            }

        if 'analytics_records_page' not in st.session_state:
            st.session_state.analytics_records_page = 1

        # 查询表单 - 修复跳转问题
        with st.form("filter_form", clear_on_submit=False):
            cols = st.columns(4)

            # 使用会话状态来保持表单值
            name_input = cols[0].text_input(
                "按姓名筛选",
                value=st.session_state.analytics_filter_params["name"],
                key="filter_name"
            )
            sn_input = cols[1].text_input(
                "按SN筛选",
                value=st.session_state.analytics_filter_params["sn"],
                key="filter_sn"
            )
            start_date_input = cols[2].date_input(
                "开始日期",
                value=st.session_state.analytics_filter_params["start_date"],
                key="filter_start_date"
            )
            end_date_input = cols[3].date_input(
                "结束日期",
                value=st.session_state.analytics_filter_params["end_date"],
                key="filter_end_date"
            )

            # 查询按钮
            submitted = st.form_submit_button("🔍 查询", use_container_width=True)

            # 处理查询提交 - 修复跳转问题
            if submitted:
                # 更新会话状态中的筛选参数
                st.session_state.analytics_filter_params = {
                    "name": name_input,
                    "sn": sn_input,
                    "start_date": start_date_input,
                    "end_date": end_date_input
                }
                # 重置到第一页
                st.session_state.analytics_records_page = 1
                # 清除缓存以确保获取新数据
                st.cache_data.clear()
                # 确保保持在当前标签页
                st.session_state.app_state['active_page'] = "检测分析"
                st.rerun()

        # 构建查询参数
        filter_params = st.session_state.analytics_filter_params
        params = {
            "page": st.session_state.analytics_records_page,
            "page_size": 10,
            "name": filter_params["name"] if filter_params["name"] else None,
            "sn": filter_params["sn"] if filter_params["sn"] else None,
            "start_date": filter_params["start_date"].isoformat() if filter_params["start_date"] else None,
            "end_date": filter_params["end_date"].isoformat() if filter_params["end_date"] else None,
        }

        @st.cache_data(ttl=10)
        def get_records(p):
            try:
                return client.get_detection_records(params={k: v for k, v in p.items() if v is not None})
            except Exception as e:
                return False, f"获取记录失败: {str(e)}"

        success, data = get_records(params)

        if not success:
            display_error_message(f"加载记录失败: {data}")
        elif data and data.get('records'):
            try:
                df = pd.DataFrame(data['records'])
                df['detected_at'] = pd.to_datetime(df['create_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

                # 修复相似度颜色显示 - 使用简单的颜色标记
                def format_similarity(val):
                    """格式化相似度显示"""
                    if pd.isna(val):
                        return ""

                    # 根据相似度值选择颜色
                    if val < 0.3:
                        color = "🔴"  # 红色
                    elif val < 0.7:
                        color = "🟡"  # 黄色
                    else:
                        color = "🟢"  # 绿色

                    return f"{color} {val:.2%}"

                # 创建显示用的DataFrame副本
                display_df = df.copy()
                display_df['similarity_display'] = display_df['similarity'].apply(format_similarity)

                # 显示DataFrame
                st.dataframe(
                    display_df,
                    column_config={
                        "image_url": st.column_config.ImageColumn("抓拍图", width="small"),
                        "name": "姓名",
                        "sn": "SN",
                        "similarity_display": st.column_config.TextColumn(
                            "相似度",
                            help="相似度值（🔴 0-30%, 🟡 30-70%, 🟢 70-100%）"
                        ),
                        "detected_at": "检测时间",
                    },
                    column_order=("image_url", "name", "sn", "similarity_display", "detected_at"),
                    hide_index=True,
                    use_container_width=True,
                    height=500
                )

                # 添加相似度颜色说明
                st.caption("🎨 相似度颜色说明: 🔴 0-30% 🟡 30-70% 🟢 70-100%")

                # 分页控制 - 修复版本，添加完整页码导航
                total_pages = data.get('total_pages', 1)
                total_records = data.get('total', 0)

                if total_pages > 1:
                    st.markdown("---")
                    st.write(f"**总计 {total_records} 条记录，共 {total_pages} 页**")

                    # 创建分页控件
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])

                    # 上一页按钮
                    with col1:
                        if st.button("⬅️ 上一页",
                                     disabled=st.session_state.analytics_records_page <= 1,
                                     use_container_width=True):
                            st.session_state.analytics_records_page -= 1
                            st.rerun()

                    # 页码导航 - 修复跳转问题
                    with col2:
                        current_page = st.session_state.analytics_records_page
                        max_visible_pages = 5

                        # 计算显示的页码范围
                        start_page = max(1, current_page - 2)
                        end_page = min(total_pages, start_page + max_visible_pages - 1)

                        if end_page - start_page + 1 < max_visible_pages:
                            start_page = max(1, end_page - max_visible_pages + 1)

                        page_buttons = st.columns(min(max_visible_pages, total_pages))

                        for i, page_num in enumerate(range(start_page, end_page + 1)):
                            with page_buttons[i]:
                                if st.button(str(page_num),
                                             type="primary" if page_num == current_page else "secondary",
                                             use_container_width=True,
                                             key=f"page_{page_num}"):
                                    if page_num != current_page:
                                        st.session_state.analytics_records_page = page_num
                                        # 确保保持在当前标签页
                                        st.session_state.app_state['active_page'] = "检测分析"
                                        st.rerun()

                    # 下一页按钮
                    with col3:
                        if st.button("下一页 ➡️",
                                     disabled=st.session_state.analytics_records_page >= total_pages,
                                     use_container_width=True):
                            st.session_state.analytics_records_page += 1
                            # 确保保持在当前标签页
                            st.session_state.app_state['active_page'] = "检测分析"
                            st.rerun()

                    # 快速跳转
                    with col4:
                        jump_page = st.number_input("跳转到",
                                                    min_value=1,
                                                    max_value=total_pages,
                                                    value=current_page,
                                                    key="jump_page_input",
                                                    label_visibility="collapsed")

                    with col5:
                        if st.button("跳转", use_container_width=True):
                            if 1 <= jump_page <= total_pages and jump_page != current_page:
                                st.session_state.analytics_records_page = jump_page
                                # 确保保持在当前标签页
                                st.session_state.app_state['active_page'] = "检测分析"
                                st.rerun()

            except Exception as e:
                display_error_message(f"记录显示失败: {str(e)}")
        else:
            st.info("在当前筛选条件下未找到任何记录。")


# 添加新的页面渲染函数
def render_statistics_page():
    """渲染统计图表页面"""
    st.header("📈 检测统计分析")
    st.markdown("---")
    client = st.session_state.app_state['api_client']

    # 数据获取函数 - 增强错误处理
    @st.cache_data(ttl=30)
    def get_analytics_data():
        """获取分析数据"""
        try:
            pie_s, pie_d = client.get_person_pie_data()
            top_s, top_d = client.get_top_persons_data(limit=10)
            hourly_s, hourly_d = client.get_hourly_trend_data()

            return {
                "pie": (pie_s, pie_d),
                "top": (top_s, top_d),
                "hourly": (hourly_s, hourly_d),
            }
        except Exception as e:
            st.error(f"获取分析数据时发生错误: {str(e)}")
            return {
                "pie": (False, "数据获取失败"),
                "top": (False, "数据获取失败"),
                "hourly": (False, "数据获取失败"),
            }

    analytics_data = get_analytics_data()

    # 统计图表页面内容
    col1, col2 = st.columns(2)

    # 人员检测分布饼图
    with col1, st.container(border=True, height=450):
        st.subheader("👥 人员检测分布")
        success, data = analytics_data["pie"]

        if not success:
            display_error_message(f"加载饼图数据失败: {data}")
            st.plotly_chart(create_empty_chart("pie", "暂无人员分布数据"), use_container_width=True)
        elif data and data.get('pie_data'):
            try:
                df = pd.DataFrame(data['pie_data'])

                # 处理小比例数据
                df.loc[df['percentage'] < 2, 'name'] = '其他'
                df = df.groupby('name')['count'].sum().reset_index()

                # 使用Plotly创建饼图
                fig = px.pie(df, values='count', names='name',
                             title="人员检测分布",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=350, showlegend=True)

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                display_error_message(f"饼图渲染失败: {str(e)}")
                st.plotly_chart(create_empty_chart("pie", "图表渲染失败"), use_container_width=True)
        else:
            st.info("暂无人员检测分布数据。")
            st.plotly_chart(create_empty_chart("pie", "暂无人员分布数据"), use_container_width=True)

    # 检测次数排行榜
    with col2, st.container(border=True, height=450):
        st.subheader("🏆 检测次数排行榜 (Top 10)")
        success, data = analytics_data["top"]

        if not success:
            display_error_message(f"加载排行榜数据失败: {data}")
            st.plotly_chart(create_empty_chart("bar", "暂无排行数据"), use_container_width=True)
        elif data and data.get('top_persons'):
            try:
                df = pd.DataFrame(data['top_persons'])

                # 使用Plotly创建水平条形图
                fig = px.bar(df, x='count', y='name', orientation='h',
                             title="检测次数排行榜",
                             labels={'count': '检测次数', 'name': '姓名'},
                             color='count',
                             color_continuous_scale='Blues')

                fig.update_layout(
                    height=350,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>检测次数: %{x}<extra></extra>'
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                display_error_message(f"排行榜渲染失败: {str(e)}")
                st.plotly_chart(create_empty_chart("bar", "图表渲染失败"), use_container_width=True)
        else:
            st.info("暂无排行数据。")
            st.plotly_chart(create_empty_chart("bar", "暂无排行数据"), use_container_width=True)

    # 24小时检测活跃度
    with st.container(border=True):
        st.subheader("🕒 24小时检测活跃度")
        success, data = analytics_data["hourly"]

        if not success:
            display_error_message(f"加载小时趋势数据失败: {data}")
            st.plotly_chart(create_empty_chart("bar", "暂无小时趋势数据"), use_container_width=True)
        elif data and data.get('hourly_data'):
            try:
                df = pd.DataFrame(data['hourly_data'])

                # 使用Plotly创建柱状图
                fig = px.bar(df, x='hour', y='count',
                             title="24小时检测活跃度",
                             labels={'hour': '小时', 'count': '检测次数'},
                             color='count',
                             color_continuous_scale='Blues')

                fig.update_layout(
                    height=300,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                    showlegend=False,
                    plot_bgcolor='rgba(255, 255, 255, 0.9)',
                    paper_bgcolor='rgba(255, 255, 255, 0.9)',
                    font=dict(color='#1a1f36')
                )
                fig.update_traces(
                    hovertemplate='<b>%{x}:00</b><br>检测次数: %{y}<extra></extra>'
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                display_error_message(f"小时趋势图渲染失败: {str(e)}")
                st.plotly_chart(create_empty_chart("bar", "图表渲染失败"), use_container_width=True)
        else:
            st.info("暂无小时趋势数据。")
            st.plotly_chart(create_empty_chart("bar", "暂无小时趋势数据"), use_container_width=True)


def render_history_page():
    """渲染历史记录页面"""
    st.header("📋 检测历史记录")
    st.markdown("---")
    client = st.session_state.app_state['api_client']

    # 历史记录查询
    st.subheader("历史检测记录查询")

    # 初始化会话状态
    if 'analytics_filter_params' not in st.session_state:
        st.session_state.analytics_filter_params = {
            "name": "",
            "sn": "",
            "start_date": None,
            "end_date": None
        }

    if 'analytics_records_page' not in st.session_state:
        st.session_state.analytics_records_page = 1

    # 查询表单
    with st.form("filter_form", clear_on_submit=False):
        cols = st.columns(4)

        # 使用会话状态来保持表单值
        name_input = cols[0].text_input(
            "按姓名筛选",
            value=st.session_state.analytics_filter_params["name"],
            key="filter_name"
        )
        sn_input = cols[1].text_input(
            "按SN筛选",
            value=st.session_state.analytics_filter_params["sn"],
            key="filter_sn"
        )
        start_date_input = cols[2].date_input(
            "开始日期",
            value=st.session_state.analytics_filter_params["start_date"],
            key="filter_start_date"
        )
        end_date_input = cols[3].date_input(
            "结束日期",
            value=st.session_state.analytics_filter_params["end_date"],
            key="filter_end_date"
        )

        # 查询按钮
        submitted = st.form_submit_button("🔍 查询", use_container_width=True)

        # 处理查询提交
        if submitted:
            # 更新会话状态中的筛选参数
            st.session_state.analytics_filter_params = {
                "name": name_input,
                "sn": sn_input,
                "start_date": start_date_input,
                "end_date": end_date_input
            }
            # 重置到第一页
            st.session_state.analytics_records_page = 1
            # 清除缓存以确保获取新数据
            st.cache_data.clear()
            # 确保保持在当前标签页
            st.session_state.app_state['active_page'] = "历史记录"
            st.rerun()

    # 构建查询参数
    filter_params = st.session_state.analytics_filter_params
    params = {
        "page": st.session_state.analytics_records_page,
        "page_size": 10,
        "name": filter_params["name"] if filter_params["name"] else None,
        "sn": filter_params["sn"] if filter_params["sn"] else None,
        "start_date": filter_params["start_date"].isoformat() if filter_params["start_date"] else None,
        "end_date": filter_params["end_date"].isoformat() if filter_params["end_date"] else None,
    }

    @st.cache_data(ttl=10)
    def get_records(p):
        try:
            return client.get_detection_records(params={k: v for k, v in p.items() if v is not None})
        except Exception as e:
            return False, f"获取记录失败: {str(e)}"

    success, data = get_records(params)

    if not success:
        display_error_message(f"加载记录失败: {data}")
    elif data and data.get('records'):
        try:
            df = pd.DataFrame(data['records'])
            df['detected_at'] = pd.to_datetime(df['create_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # 修复相似度颜色显示
            def format_similarity(val):
                """格式化相似度显示"""
                if pd.isna(val):
                    return ""

                # 根据相似度值选择颜色
                if val < 0.3:
                    color = "🔴"  # 红色
                elif val < 0.7:
                    color = "🟡"  # 黄色
                else:
                    color = "🟢"  # 绿色

                return f"{color} {val:.2%}"

            # 创建显示用的DataFrame副本
            display_df = df.copy()
            display_df['similarity_display'] = display_df['similarity'].apply(format_similarity)

            # 显示DataFrame
            st.dataframe(
                display_df,
                column_config={
                    "image_url": st.column_config.ImageColumn("抓拍图", width="small"),
                    "name": "姓名",
                    "sn": "SN",
                    "similarity_display": st.column_config.TextColumn(
                        "相似度",
                        help="相似度值（🔴 0-30%, 🟡 30-70%, 🟢 70-100%）"
                    ),
                    "detected_at": "检测时间",
                },
                column_order=("image_url", "name", "sn", "similarity_display", "detected_at"),
                hide_index=True,
                use_container_width=True,
                height=500
            )

            # 添加相似度颜色说明
            st.caption("🎨 相似度颜色说明: 🔴 0-30% 🟡 30-70% 🟢 70-100%")

            # 分页控制
            total_pages = data.get('total_pages', 1)
            total_records = data.get('total', 0)

            if total_pages > 1:
                st.markdown("---")
                st.write(f"**总计 {total_records} 条记录，共 {total_pages} 页**")

                # 创建分页控件
                col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])

                # 上一页按钮
                with col1:
                    if st.button("⬅️ 上一页",
                                 disabled=st.session_state.analytics_records_page <= 1,
                                 use_container_width=True):
                        st.session_state.analytics_records_page -= 1
                        st.rerun()

                # 页码导航
                with col2:
                    current_page = st.session_state.analytics_records_page
                    max_visible_pages = 5

                    # 计算显示的页码范围
                    start_page = max(1, current_page - 2)
                    end_page = min(total_pages, start_page + max_visible_pages - 1)

                    if end_page - start_page + 1 < max_visible_pages:
                        start_page = max(1, end_page - max_visible_pages + 1)

                    page_buttons = st.columns(min(max_visible_pages, total_pages))

                    for i, page_num in enumerate(range(start_page, end_page + 1)):
                        with page_buttons[i]:
                            if st.button(str(page_num),
                                         type="primary" if page_num == current_page else "secondary",
                                         use_container_width=True,
                                         key=f"page_{page_num}"):
                                if page_num != current_page:
                                    st.session_state.analytics_records_page = page_num
                                    # 确保保持在当前标签页
                                    st.session_state.app_state['active_page'] = "历史记录"
                                    st.rerun()

                # 下一页按钮
                with col3:
                    if st.button("下一页 ➡️",
                                 disabled=st.session_state.analytics_records_page >= total_pages,
                                 use_container_width=True):
                        st.session_state.analytics_records_page += 1
                        # 确保保持在当前标签页
                        st.session_state.app_state['active_page'] = "历史记录"
                        st.rerun()

                # 快速跳转
                with col4:
                    jump_page = st.number_input("跳转到",
                                                min_value=1,
                                                max_value=total_pages,
                                                value=current_page,
                                                key="jump_page_input",
                                                label_visibility="collapsed")

                with col5:
                    if st.button("跳转", use_container_width=True):
                        if 1 <= jump_page <= total_pages and jump_page != current_page:
                            st.session_state.analytics_records_page = jump_page
                            # 确保保持在当前标签页
                            st.session_state.app_state['active_page'] = "历史记录"
                            st.rerun()

        except Exception as e:
            display_error_message(f"记录显示失败: {str(e)}")
    else:
        st.info("在当前筛选条件下未找到任何记录。")


# ==============================================================================
# 6. 主程序入口 (Main Application)
# ==============================================================================
def main():
    """主程序入口"""
    try:
        initialize_session_state()
        render_sidebar()

        main_content = st.container()

        with main_content:
            # API连接状态检查
            if not st.session_state.app_state['api_status'][0]:
                st.warning("⚠️ API服务未连接，请在左侧侧边栏配置正确的服务地址并确保后端服务已启动。页面功能将受限。")
                st.info("💡 提示：默认服务地址为 localhost:12010，请根据实际情况修改。")

            # 检查是否需要刷新数据
            if st.session_state.data_refresh_trigger:
                st.cache_data.clear()
                st.session_state.data_refresh_trigger = False
                st.toast("数据已刷新", icon="✅")

            # 页面路由
            page_map = {
                "数据看板": render_dashboard_page,
                "人脸库管理": render_management_page,
                "实时监测": render_monitoring_page,
                "统计图表": render_statistics_page,
                "历史记录": render_history_page,
            }

            active_page_func = page_map.get(st.session_state.app_state['active_page'])
            if active_page_func:
                try:
                    active_page_func()
                except Exception as e:
                    st.error(f"❌ 页面渲染失败: {str(e)}")
                    st.info("💡 请尝试刷新页面或检查网络连接。如果问题持续存在，请联系系统管理员。")

                    # 显示详细错误信息（仅在开发模式下）
                    if st.session_state.app_state.get('api_url', '').startswith('localhost'):
                        with st.expander("🔍 错误详情（开发模式）"):
                            st.code(str(e))
            else:
                st.error("❌ 未知页面")
                st.info("💡 请检查页面配置或联系系统管理员。")

    except Exception as e:
        st.error(f"❌ 应用程序启动失败: {str(e)}")
        st.info("💡 请检查配置并重新启动应用。如果问题持续存在，请查看日志文件。")

        # 显示启动错误详情
        with st.expander("🔍 启动错误详情"):
            st.code(str(e))


if __name__ == "__main__":
    main()