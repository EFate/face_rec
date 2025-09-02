# 全面修复后的 ui.py 代码
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
    html, body, [class*="st-"] {
        font-family: 'Noto Sans SC', sans-serif;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1a1f36;
        font-weight: 700;
    }

    /* --- 侧边栏 --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4e8;
        box-shadow: 2px 0px 10px rgba(0,0,0,0.05);
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
        padding: 10px 15px !important;
        margin-bottom: 8px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .nav-item button[kind="secondary"] {
        background-color: transparent;
        color: #333;
        border: none;
    }
    .nav-item button[kind="secondary"]:hover {
        background-color: #f0f2f6;
    }
    .nav-item button[kind="primary"] {
        background-color: #4f46e5;
        color: white;
        border: none;
    }
    .nav-item button span {
        margin-right: 12px;
    }

    /* --- 指标卡片 (Metric Card) --- */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .metric-card .title {
        font-size: 1rem;
        font-weight: 500;
        color: #6c757d;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1f36;
        line-height: 1.2;
    }
    .metric-card.ok { border-left: 5px solid #28a745; }
    .metric-card.error { border-left: 5px solid #dc3545; }

    /* --- 通用按钮美化 --- */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:not([kind="secondary"]) {
        border: 1px solid #4f46e5;
        background-color: #4f46e5;
        color: white;
    }
    .stButton>button:not([kind="secondary"]):hover {
        background-color: #4338ca;
        border-color: #4338ca;
    }
    .stButton>button[kind="secondary"][aria-label*="删除"] {
        background-color: #f5f5f5;
        border: 1px solid #dc3545;
        color: #dc3545;
    }
    .stButton>button[kind="secondary"][aria-label*="删除"]:hover {
        background-color: #dc3545;
        color: white;
    }
    
    /* --- 容器和 Expander --- */
    .st-emotion-cache-1r6slb0 {
        border-radius: 12px;
        border: 1px solid #e0e4e8;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    [data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #e0e4e8 !important;
    }

    /* --- 标签页 (Tabs) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0 0.5rem;
        height: 50px;
        background-color: transparent;
        border-bottom: 4px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #4f46e5;
        color: #4f46e5;
        font-weight: 600;
    }

    /* --- 错误和成功消息样式 --- */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }

</style>
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
            'streams_stop': '/api/face/streams/stop/{}',
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
    
    def start_stream(self, source, lifetime): 
        return self._request('POST', 'streams_start', 
                           json={"source": source, "lifetime_minutes": lifetime})
    
    def stop_stream(self, stream_id): 
        return self._request('POST', 'streams_stop', 
                           url_format=self.endpoints['streams_stop'].format(stream_id))
    
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


# ==============================================================================
# 5. UI 渲染模块 (全面修复和美化)
# ==============================================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("✨ 人脸识别系统")
        st.caption("v3.2 - Enhanced & Fixed")

        # API连接配置
        api_url = st.text_input("后端服务地址", value=st.session_state.app_state['api_url'])
        if api_url != st.session_state.app_state['api_url']:
            st.session_state.app_state['api_url'] = api_url
            st.session_state.app_state['api_client'] = ApiClient(api_url)
            st.rerun()

        # 健康检查
        client = st.session_state.app_state['api_client']
        success, data = client.check_health()
        
        if success:
            status_msg = data.get('message', "连接成功") if isinstance(data, dict) else "连接成功"
            status_icon = "✅"
        else:
            status_msg = str(data)
            status_icon = "❌"
            
        st.session_state.app_state['api_status'] = (success, status_msg)
        st.info(f"**API状态:** {status_msg}", icon=status_icon)
        
        st.divider()

        # 导航菜单
        st.markdown("<h6>导航</h6>", unsafe_allow_html=True)
        pages = {
            "数据看板": "📊",
            "人脸库管理": "🗂️",
            "实时监测": "🛰️",
            "检测分析": "🔍"
        }
        
        for page, icon in pages.items():
            st.markdown(f'<div class="nav-item">', unsafe_allow_html=True)
            if st.button(
                f"{icon} {page}",
                use_container_width=True,
                type="primary" if st.session_state.app_state['active_page'] == page else "secondary"
            ):
                st.session_state.app_state['active_page'] = page
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 强制刷新全站数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_dashboard_page():
    """渲染数据看板页面 - 修复版本"""
    st.header("📊 数据看板总览")
    client = st.session_state.app_state['api_client']

    @st.cache_data(ttl=30)
    def get_dashboard_data():
        """获取看板数据"""
        try:
            stats_s, stats_d = client.get_detection_stats()
            faces_s, faces_d = client.get_all_faces()
            trend_s, trend_d = client.get_weekly_trend()
            
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
            st.error(f"获取看板数据时发生错误: {str(e)}")
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

    # 指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        count = data.get('faces', {}).get('count', 'N/A')
        st.html(f"""<div class="metric-card"><div class="title">人脸库人员总数</div><div class="value">{count}</div></div>""")
    with col2: 
        total = data.get('stats', {}).get('total_detections', 'N/A')
        st.html(f"""<div class="metric-card"><div class="title">总检测次数</div><div class="value">{total}</div></div>""")
    with col3: 
        today = data.get('stats', {}).get('today_detections', 'N/A')
        st.html(f"""<div class="metric-card"><div class="title">今日检测</div><div class="value">{today}</div></div>""")
    with col4: 
        st.html(f"""<div class="metric-card {api_color_class}"><div class="title">API 服务</div><div class="value">{api_status}</div></div>""")

    st.markdown("<br>", unsafe_allow_html=True)

    # 趋势图表和最新记录
    col1, col2 = st.columns([0.65, 0.35])
    
    with col1, st.container(border=True):
        st.subheader("🗓️ 近7日检测趋势")
        trend_data = data.get('trend', {}).get('trend_data', [])
        
        if trend_data:
            try:
                trend_df = pd.DataFrame(trend_data)
                trend_df['date'] = pd.to_datetime(trend_df['date'])
                
                chart = alt.Chart(trend_df).mark_area(
                    line={'color':'#4f46e5'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[
                            alt.GradientStop(color='#4f46e5', offset=0), 
                            alt.GradientStop(color='white', offset=1)
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('date:T', title='日期', axis=alt.Axis(format='%m-%d', labelAngle=0)),
                    y=alt.Y('count:Q', title='检测次数', axis=alt.Axis(grid=True)),
                    tooltip=[
                        alt.Tooltip('date:T', title='日期'), 
                        alt.Tooltip('count:Q', title='次数')
                    ]
                ).properties(height=350).interactive()
                
                st.altair_chart(chart, use_container_width=True)
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
                for item in recent:
                    col_img, col_info = st.columns([0.3, 0.7])
                    
                    # 安全地显示图片
                    try:
                        if item.get('image_url'):
                            col_img.image(item['image_url'], width=60)
                        else:
                            col_img.write("无图片")
                    except:
                        col_img.write("图片加载失败")
                    
                    # 显示信息
                    name = item.get('name', 'Unknown')
                    col_info.markdown(f"**{name}**")
                    
                    # 安全地格式化时间
                    time_str = safe_format_datetime(item.get('create_time', ''))
                    col_info.caption(f"{time_str}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"最新记录显示失败: {str(e)}")
        else: 
            st.info("暂无最近检测记录。")
    
    st.divider()

    # 快速人脸识别 - 修复版本
    with st.expander("🧐 快速人脸识别", expanded=True):
        uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img_col, res_col = st.columns(2)
            
            # 显示上传的图片
            try:
                img_col.image(uploaded_file, caption="待识别图片", width=300)
            except Exception as e:
                img_col.error(f"图片显示失败: {str(e)}")
            
            # 识别按钮和结果显示
            if res_col.button("🔍 开始识别", type="primary", use_container_width=True):
                with res_col:
                    with st.spinner("正在识别中，请稍候..."):
                        try:
                            files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                            success, results = client.recognize_face(files)
                            
                            if success:
                                if isinstance(results, list) and len(results) > 0:
                                    st.success(f"🎉 识别成功！找到 {len(results)} 个匹配项")
                                    
                                    for i, res in enumerate(results, 1):
                                        with st.container():
                                            st.markdown(f"**匹配结果 {i}:**")
                                            st.info(f"""
                                            **姓名:** {res.get('name', 'Unknown')}  
                                            **SN:** {res.get('sn', 'Unknown')}  
                                            **相似度:** {res.get('similarity', 0):.2%}  
                                            **检测置信度:** {res.get('detection_confidence', 0):.2%}
                                            """)
                                else:
                                    st.warning("🤔 检测到人脸，但未在库中找到匹配项")
                            else:
                                display_error_message(f"识别失败: {results}")
                                
                        except Exception as e:
                            display_error_message(f"识别过程中发生错误: {str(e)}")


def render_management_page():
    """渲染人脸库管理页面"""
    st.header("🗂️ 人脸库管理中心")
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
    """渲染实时监测页面"""
    st.header("🛰️ 实时视频监测")
    client = st.session_state.app_state['api_client']

    with st.expander("▶️ 启动新监测任务", expanded=True):
        with st.form("start_stream_form"):
            source = st.text_input("视频源", "0", help="摄像头ID(0, 1) 或 视频文件/URL")
            lifetime = st.number_input("生命周期(分钟)", min_value=-1, value=10, help="-1 代表永久")
            
            if st.form_submit_button("🚀 开启监测", use_container_width=True, type="primary"):
                with st.spinner("请求启动视频流..."):
                    try:
                        success, data = client.start_stream(source, lifetime)
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
            st.subheader(f"正在播放: `{stream_info['source']}`")
            st.caption(f"Stream ID: `{stream_info['stream_id']}`")
            try:
                st.image(stream_info['feed_url'])
            except Exception as e:
                st.error(f"视频流加载失败: {str(e)}")
    else: 
        st.info("请从下方列表选择一个流进行观看，或启动一个新任务。")
    
    st.divider()

    # 活动流列表
    st.subheader("所有活动中的监测任务")
    
    @st.cache_data(ttl=5)
    def get_active_streams():
        try:
            success, data = client.list_streams()
            return data.get('streams', []) if success else []
        except Exception as e:
            st.error(f"获取活动流失败: {str(e)}")
            return []

    active_streams = get_active_streams()
    
    if not active_streams: 
        st.info("目前没有正在运行的视频监测任务。")
    else:
        for stream in active_streams:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    try:
                        expires_at = datetime.fromisoformat(stream['expires_at']) if stream.get('expires_at') else None
                        expires_display = expires_at.strftime('%Y-%m-%d %H:%M:%S') if expires_at else "永久"
                        st.markdown(f"**来源:** `{stream['source']}` | **过期时间:** {expires_display}")
                        st.caption(f"ID: `{stream['stream_id']}`")
                    except Exception as e:
                        st.warning(f"流信息显示异常: {str(e)}")
                
                with col2:
                    btn_cols = st.columns(2)
                    
                    if btn_cols[0].button("👁️", key=f"view_{stream['stream_id']}", 
                                        help="观看此流", use_container_width=True):
                        st.session_state.app_state['monitoring']['viewing_stream_info'] = stream
                        st.rerun()
                    
                    if btn_cols[1].button("⏹️", key=f"stop_{stream['stream_id']}", 
                                        help="停止此流", type="secondary", use_container_width=True):
                        with st.spinner("停止中..."):
                            try:
                                success, _ = client.stop_stream(stream['stream_id'])
                                if success:
                                    st.toast("视频流已停止。", icon="✅")
                                    if stream_info and stream_info['stream_id'] == stream['stream_id']:
                                        st.session_state.app_state['monitoring']['viewing_stream_info'] = None
                                    st.rerun()
                                else:
                                    st.error("停止失败")
                            except Exception as e:
                                display_error_message(f"停止过程中发生错误: {str(e)}")


def render_analytics_page():
    """渲染检测分析页面 - 全面修复版本"""
    st.header("🔍 检测分析中心")
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
                    fig.update_layout(height=350, showlegend=True)
                    
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

    # 历史记录查询 - 修复版本
    with tab2:
        st.subheader("历史检测记录查询")
        
        with st.form("filter_form"):
            cols = st.columns(4)
            name = cols[0].text_input("按姓名筛选")
            sn = cols[1].text_input("按SN筛选")
            start_date = cols[2].date_input("开始日期", value=None)
            end_date = cols[3].date_input("结束日期", value=None)
            submitted = st.form_submit_button("🔍 查询", use_container_width=True)

        # 分页控制
        if 'analytics_records_page' not in st.session_state:
            st.session_state.analytics_records_page = 1

        params = {
            "page": st.session_state.analytics_records_page,
            "page_size": 10,
            "name": name if name else None,
            "sn": sn if sn else None,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
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
                
                st.dataframe(
                    df,
                    column_config={
                        "image_url": st.column_config.ImageColumn("抓拍图", width="small"),
                        "name": "姓名", 
                        "sn": "SN",
                        "similarity": st.column_config.ProgressColumn(
                            "相似度", format="%.2f", min_value=0, max_value=1
                        ),
                        "detected_at": "检测时间",
                    },
                    column_order=("image_url", "name", "sn", "similarity", "detected_at"),
                    hide_index=True, 
                    use_container_width=True, 
                    height=500
                )
                
                # 分页控制
                total_pages = data.get('total_pages', 1)
                if total_pages > 1:
                    page_cols = st.columns([0.6, 0.2, 0.2])
                    page_cols[0].write(f"总计 {data.get('total')} 条记录，共 {total_pages} 页")
                    
                    # 上一页按钮
                    if page_cols[1].button("⬅️ 上一页", disabled=st.session_state.analytics_records_page <= 1):
                        st.session_state.analytics_records_page -= 1
                        st.rerun()
                    
                    # 下一页按钮
                    if page_cols[2].button("下一页 ➡️", disabled=st.session_state.analytics_records_page >= total_pages):
                        st.session_state.analytics_records_page += 1
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

            # 页面路由
            page_map = {
                "数据看板": render_dashboard_page,
                "人脸库管理": render_management_page,
                "实时监测": render_monitoring_page,
                "检测分析": render_analytics_page,
            }
            
            active_page_func = page_map.get(st.session_state.app_state['active_page'])
            if active_page_func:
                try:
                    active_page_func()
                except Exception as e:
                    st.error(f"页面渲染失败: {str(e)}")
                    st.info("请尝试刷新页面或检查网络连接。")
            else:
                st.error("未知页面")
                
    except Exception as e:
        st.error(f"应用程序启动失败: {str(e)}")
        st.info("请检查配置并重新启动应用。")


if __name__ == "__main__":
    main()
