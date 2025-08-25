"""
人脸识别管理系统 - 优化版UI
功能完整、界面美观、逻辑清晰的人脸识别管理系统
"""

import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any, Dict, List, Optional
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
import os
import altair as alt
from PIL import Image
import io

# ==============================================================================
# 1. 页面配置与主题设置
# ==============================================================================

st.set_page_config(
    page_title="人脸识别管理系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '人脸识别管理系统 v2.0 - 功能完整的智能识别平台'
    }
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 卡片样式 */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e6ed;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    /* 表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 2px solid #e0e6ed;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #7f8c8d;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea !important;
        border-bottom: 3px solid #667eea;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div {
        border-radius: 10px;
        border: 1px solid #e0e6ed;
    }
    
    /* 成功状态颜色 */
    .success-text {
        color: #27ae60;
        font-weight: bold;
    }
    
    .warning-text {
        color: #f39c12;
        font-weight: bold;
    }
    
    .error-text {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 全局配置与状态管理
# ==============================================================================

@st.cache_data(ttl=60)
def get_api_config():
    """获取API配置"""
    backend_host = os.getenv("HOST__IP", "localhost")
    backend_port = os.getenv("SERVER__PORT", "12010")
    return f"{backend_host}:{backend_port}"

# API端点配置
API_ENDPOINTS = {
    # 系统健康检查
    'HEALTH': '/api/face/health',
    
    # 人脸管理
    'FACES': '/api/face/faces',
    'FACE_BY_SN': '/api/face/faces/{}',
    'REGISTER_FACE': '/api/face/faces',
    'UPDATE_FACE': '/api/face/faces/{}',
    'DELETE_FACE': '/api/face/faces/{}',
    'RECOGNIZE': '/api/face/recognize',
    
    # 视频流管理
    'STREAMS_START': '/api/face/streams/start',
    'STREAMS_STOP': '/api/face/streams/stop/{}',
    'STREAMS_LIST': '/api/face/streams',
    'STREAM_FEED': '/api/face/streams/feed/{}',
    
    # 检测记录
    'DETECTION_RECORDS': '/api/detection/records',
    'DETECTION_STATS': '/api/detection/stats',
    'DETECTION_RECORD_DETAIL': '/api/detection/records/{}',
    'DETECTION_WEEKLY_TREND': '/api/detection/weekly-trend',
    'DETECTION_PERSON_PIE': '/api/detection/person-pie',
    'DETECTION_HOURLY_TREND': '/api/detection/hourly-trend',
    'DETECTION_TOP_PERSONS': '/api/detection/top-persons',
}

def initialize_session_state():
    """初始化会话状态"""
    defaults = {
        'api_url': get_api_config(),
        'api_status': (False, '未连接'),
        'current_page': '仪表盘',
        'faces_data': None,
        'detection_stats': None,
        'detection_records': None,
        'weekly_trend': None,
        'person_pie_data': None,
        'hourly_trend': None,
        'top_persons': None,
        'active_streams': [],
        'selected_stream': None,
        'detection_page': 1,
        'detection_page_size': 20,
        'filters': {
            'name': '',
            'sn': '',
            'start_date': None,
            'end_date': None
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==============================================================================
# 3. API通信工具函数
# ==============================================================================

@st.cache_data(ttl=30)
def check_api_status(api_url: str) -> Tuple[bool, str]:
    """检查API健康状态"""
    try:
        url = f"http://{api_url}{API_ENDPOINTS['HEALTH']}"
        response = requests.get(url, timeout=5)
        if response.ok:
            return True, "服务运行正常"
        return False, f"服务异常 (HTTP {response.status_code})"
    except requests.exceptions.RequestException as e:
        return False, f"连接失败: {str(e)}"

def parse_error_response(response: requests.Response) -> str:
    """解析错误响应"""
    try:
        error_data = response.json()
        if "detail" in error_data:
            detail = error_data["detail"]
            if isinstance(detail, list):
                return "; ".join([f"{d.get('loc', [''])[-1]}: {d.get('msg', '')}" for d in detail])
            return str(detail)
        return error_data.get("msg", "未知错误")
    except:
        return f"HTTP {response.status_code}: {response.text}"

def make_api_request(method: str, endpoint: str, **kwargs) -> Tuple[bool, Any, str]:
    """统一的API请求函数"""
    try:
        url = f"http://{st.session_state.api_url}{endpoint}"
        response = requests.request(method, url, timeout=30, **kwargs)
        
        if response.ok:
            try:
                data = response.json()
                if data.get("code") == 0:
                    return True, data.get("data"), data.get("msg", "操作成功")
                else:
                    return False, None, data.get("msg", "操作失败")
            except json.JSONDecodeError:
                return True, None, "操作成功"
        else:
            return False, None, parse_error_response(response)
    except requests.exceptions.RequestException as e:
        return False, None, f"网络错误: {str(e)}"

# ==============================================================================
# 4. 数据加载函数
# ==============================================================================

def refresh_all_data():
    """刷新所有数据"""
    with st.spinner("正在刷新数据..."):
        # 清除缓存
        st.cache_data.clear()
        
        # 重新加载所有数据
        load_faces_data()
        load_detection_stats()
        load_detection_records()
        load_charts_data()
        load_active_streams()
        
        st.toast("数据刷新完成！", icon="✅")

@st.cache_data(ttl=60)
def load_faces_data():
    """加载人脸数据"""
    success, data, msg = make_api_request('GET', API_ENDPOINTS['FACES'])
    if success and data:
        faces = data.get('faces', [])
        unique_sns = list(set(face['sn'] for face in faces))
        return {
            'count': len(faces),
            'persons': {sn: [f for f in faces if f['sn'] == sn] for sn in unique_sns},
            'all_faces': faces
        }
    return {'count': 0, 'persons': {}, 'all_faces': []}

@st.cache_data(ttl=30)
def load_detection_stats():
    """加载检测统计"""
    success, data, msg = make_api_request('GET', API_ENDPOINTS['DETECTION_STATS'])
    if success and data:
        return data
    return None

@st.cache_data(ttl=30)
def load_detection_records(page=1, page_size=20, **filters):
    """加载检测记录"""
    params = {'page': page, 'page_size': page_size}
    
    # 添加过滤条件
    for key, value in filters.items():
        if value:
            params[key] = value
    
    success, data, msg = make_api_request('GET', API_ENDPOINTS['DETECTION_RECORDS'], params=params)
    return data if success else None

@st.cache_data(ttl=60)
def load_charts_data():
    """加载图表数据"""
    # 周趋势
    success, weekly_data, _ = make_api_request('GET', API_ENDPOINTS['DETECTION_WEEKLY_TREND'])
    if success:
        st.session_state.weekly_trend = weekly_data
    
    # 人员分布
    success, pie_data, _ = make_api_request('GET', API_ENDPOINTS['DETECTION_PERSON_PIE'])
    if success:
        st.session_state.person_pie_data = pie_data
    
    # 小时分布
    success, hourly_data, _ = make_api_request('GET', API_ENDPOINTS['DETECTION_HOURLY_TREND'])
    if success:
        st.session_state.hourly_trend = hourly_data
    
    # 排行榜
    success, top_data, _ = make_api_request('GET', API_ENDPOINTS['DETECTION_TOP_PERSONS'], params={'limit': 10})
    if success:
        st.session_state.top_persons = top_data

@st.cache_data(ttl=10)
def load_active_streams():
    """加载活动视频流"""
    success, data, msg = make_api_request('GET', API_ENDPOINTS['STREAMS_LIST'])
    if success and data:
        return data.get('streams', [])
    return []

# ==============================================================================
# 5. 工具函数
# ==============================================================================

def format_datetime(dt_str: str) -> str:
    """格式化日期时间"""
    if not dt_str:
        return "永久"
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str

def format_image_url(image_path: str) -> str:
    """格式化图片URL"""
    if not image_path:
        return "https://via.placeholder.com/300x200?text=No+Image"
    
    if image_path.startswith('http'):
        return image_path
    
    if image_path.startswith('/data/'):
        return f"http://{st.session_state.api_url}{image_path}"
    
    return f"http://{st.session_state.api_url}/data/detected_imgs/{image_path}"

def display_image_with_fallback(image_url: str, caption: str = "", width: int = 300):
    """显示图片，带错误处理"""
    try:
        st.image(image_url, caption=caption, width=width)
    except:
        st.image("https://via.placeholder.com/300x200?text=Image+Error", caption="图片加载失败")

# ==============================================================================
# 6. 侧边栏组件
# ==============================================================================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🤖 人脸识别系统")
        st.markdown("*智能识别 • 高效管理*")
        
        st.divider()
        
        # API配置
        st.subheader("🔧 系统配置")
        new_api_url = st.text_input(
            "后端地址",
            value=st.session_state.api_url,
            help="格式: IP:端口 (如 192.168.1.100:12010)"
        )
        
        if new_api_url != st.session_state.api_url:
            st.session_state.api_url = new_api_url
            st.rerun()
        
        # 检查API状态
        is_connected, status_msg = check_api_status(st.session_state.api_url)
        st.session_state.api_status = (is_connected, status_msg)
        
        status_color = "#27ae60" if is_connected else "#e74c3c"
        st.markdown(f"""
        <div style="padding: 10px; background: {status_color}20; border-radius: 10px; border-left: 4px solid {status_color};">
            <strong>API状态:</strong> {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # 导航菜单
        st.subheader("🧭 功能导航")
        
        pages = {
            "仪表盘": "📊",
            "人脸库管理": "👥",
            "实时监测": "📹",
            "检测记录": "📝",
            "统计分析": "📈"
        }
        
        for page_name, icon in pages.items():
            if st.button(
                f"{icon} {page_name}",
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_name else "secondary"
            ):
                st.session_state.current_page = page_name
                st.rerun()
        
        st.divider()
        
        # 快捷操作
        st.subheader("⚡ 快捷操作")
        
        if st.button("🔄 刷新全部数据", use_container_width=True):
            refresh_all_data()
            st.rerun()
        
        if st.button("💾 导出报告", use_container_width=True, disabled=not is_connected):
            st.info("报告导出功能开发中...")
        
        st.divider()
        
        # 系统信息
        st.markdown("""
        <div style="font-size: 0.8em; color: #7f8c8d;">
            <strong>版本:</strong> v2.0<br>
            <strong>作者:</strong> AI团队<br>
            <strong>更新:</strong> 2024
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 7. 页面组件
# ==============================================================================

def render_dashboard_page():
    """仪表盘页面"""
    st.title("📊 系统仪表盘")
    
    if not st.session_state.api_status[0]:
        st.error("⚠️ API服务未连接，请在侧边栏配置正确的服务地址")
        return
    
    # 加载数据
    faces_data = load_faces_data()
    stats_data = load_detection_stats()
    active_streams = load_active_streams()
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">{len(faces_data['persons'])}</div>
            <div class="metric-label">注册人员</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📸</div>
            <div class="metric-value">{faces_data['count']}</div>
            <div class="metric-label">人脸图片</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔢</div>
            <div class="metric-value">{stats_data.get('total_detections', 0) if stats_data else 0}</div>
            <div class="metric-label">总检测次数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📹</div>
            <div class="metric-value">{len(active_streams)}</div>
            <div class="metric-label">活动视频流</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 快速操作区域
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 快速识别")
        uploaded_file = st.file_uploader(
            "上传图片进行识别",
            type=['jpg', 'jpeg', 'png'],
            key="quick_recognize"
        )
        
        if uploaded_file:
            col_img, col_res = st.columns([1, 1])
            
            with col_img:
                st.image(uploaded_file, caption="待识别图片", use_container_width=True)
            
            with col_res:
                if st.button("开始识别", type="primary"):
                    with st.spinner("正在识别..."):
                        files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                        success, results, msg = make_api_request('POST', API_ENDPOINTS['RECOGNIZE'], files=files)
                        
                        if success:
                            if results:
                                st.success(f"识别成功！找到 {len(results)} 个匹配")
                                for result in results:
                                    similarity = result.get('similarity', 0) * 100
                                    st.markdown(f"""
                                    **{result.get('name')}** (SN: {result.get('sn')})  
                                    相似度: <span class="success-text">{similarity:.1f}%</span>
                                    """)
                            else:
                                st.info("未找到匹配的人脸")
                        else:
                            st.error(f"识别失败: {msg}")
    
    with col2:
        st.subheader("📈 今日统计")
        if stats_data:
            today = stats_data.get('today_detections', 0)
            unique = stats_data.get('unique_persons', 0)
            
            st.metric("今日检测", today)
            st.metric("今日人员", unique)
            
            # 显示最近检测
            recent = stats_data.get('recent_detections', [])[:5]
            if recent:
                st.subheader("最近检测")
                for det in recent:
                    st.markdown(f"""
                    **{det.get('name')}**  
                    <small>{format_datetime(det.get('detected_at'))}</small>
                    """)
    
    st.divider()
    
    # 图表区域
    load_charts_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 7天趋势")
        if st.session_state.get('weekly_trend'):
            data = st.session_state.weekly_trend
            df = pd.DataFrame({
                '日期': data.get('dates', []),
                '检测次数': data.get('counts', [])
            })
            
            chart = alt.Chart(df).mark_line(point=True).encode(
                x='日期',
                y='检测次数',
                tooltip=['日期', '检测次数']
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("暂无趋势数据")
    
    with col2:
        st.subheader("🏆 检测排行")
        if st.session_state.get('top_persons'):
            data = st.session_state.top_persons
            persons = data.get('persons', [])
            
            for i, person in enumerate(persons[:5]):
                emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                st.markdown(f"{emoji} **{person.get('name')}** - {person.get('count')}次")

# ==============================================================================
# 8. 人脸库管理页面
# ==============================================================================

def render_faces_management_page():
    """人脸库管理页面"""
    st.title("👥 人脸库管理")
    
    if not st.session_state.api_status[0]:
        st.error("⚠️ API服务未连接")
        return
    
    # 注册新人员
    with st.expander("➕ 注册新人员", expanded=False):
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("姓名 *", placeholder="请输入人员姓名")
                sn = st.text_input("编号(SN) *", placeholder="请输入唯一编号")
            
            with col2:
                uploaded_files = st.file_uploader(
                    "上传人脸照片 *",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    help="可上传多张图片"
                )
            
            if st.form_submit_button("注册人员", type="primary"):
                if not all([name, sn, uploaded_files]):
                    st.error("请填写所有必填项")
                else:
                    success_count = 0
                    for file in uploaded_files:
                        files = {'image_file': (file.name, file.getvalue())}
                        data = {'name': name, 'sn': sn}
                        
                        success, result, msg = make_api_request(
                            'POST',
                            API_ENDPOINTS['REGISTER_FACE'],
                            data=data,
                            files=files
                        )
                        
                        if success:
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"成功注册 {success_count} 张人脸图片！")
                        st.rerun()
                    else:
                        st.error("注册失败")
    
    st.divider()
    
    # 加载人脸数据
    faces_data = load_faces_data()
    
    if not faces_data['persons']:
        st.info("人脸库为空，请先注册人员")
        return
    
    # 搜索和筛选
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 搜索人员", placeholder="输入姓名或SN搜索")
    with col2:
        sort_by = st.selectbox("排序方式", ["姓名", "SN", "图片数量"])
    
    # 筛选人员
    filtered_persons = {}
    for sn, faces in faces_data['persons'].items():
        if search_term:
            if search_term.lower() not in faces[0]['name'].lower() and search_term.lower() not in sn.lower():
                continue
        filtered_persons[sn] = faces
    
    st.subheader(f"共找到 {len(filtered_persons)} 位人员")
    
    # 显示人员列表
    cols = st.columns(3)
    for idx, (sn, faces) in enumerate(filtered_persons.items()):
        col = cols[idx % 3]
        
        with col:
            with st.container():
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 5px 0;">
                    <h4>{faces[0]['name']}</h4>
                    <p><strong>SN:</strong> {sn}</p>
                    <p><strong>图片数:</strong> {len(faces)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 图片预览
                if faces:
                    image_urls = [format_image_url(f['image_path']) for f in faces[:3]]
                    st.image(image_urls, width=100, caption=[f"图片{i+1}" for i in range(len(image_urls))])
                
                # 操作按钮
                col_edit, col_delete = st.columns(2)
                
                with col_edit:
                    if st.button("✏️ 编辑", key=f"edit_{sn}"):
                        new_name = st.text_input("新姓名", value=faces[0]['name'], key=f"new_name_{sn}")
                        if st.button("更新", key=f"update_{sn}"):
                            endpoint = API_ENDPOINTS['UPDATE_FACE'].format(sn)
                            success, _, msg = make_api_request('PUT', endpoint, json={'name': new_name})
                            if success:
                                st.success("更新成功！")
                                st.rerun()
                
                with col_delete:
                    if st.button("🗑️ 删除", key=f"delete_{sn}"):
                        if st.checkbox(f"确认删除 {faces[0]['name']}？"):
                            endpoint = API_ENDPOINTS['DELETE_FACE'].format(sn)
                            success, _, msg = make_api_request('DELETE', endpoint)
                            if success:
                                st.success("删除成功！")
                                st.rerun()

# ==============================================================================
# 9. 实时监测页面
# ==============================================================================

def render_monitoring_page():
    """实时监测页面"""
    st.title("📹 实时视频监测")
    
    if not st.session_state.api_status[0]:
        st.error("⚠️ API服务未连接")
        return
    
    # 启动新视频流
    with st.expander("🚀 启动新监测", expanded=True):
        with st.form("start_stream_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                source = st.text_input("视频源", value="0", 
                                     help="摄像头ID(0,1,2...) 或 RTSP/HTTP流地址")
                lifetime = st.number_input("运行时长(分钟)", value=60, min_value=-1, 
                                         help="-1为永久运行")
            
            with col2:
                st.markdown("""
                **示例:**
                - `0` - 默认摄像头
                - `rtsp://user:pass@ip:port/stream`
                - `http://example.com/video.mp4`
                """)
            
            if st.form_submit_button("启动监测", type="primary"):
                payload = {'source': source, 'lifetime_minutes': lifetime}
                success, data, msg = make_api_request('POST', API_ENDPOINTS['STREAMS_START'], json=payload)
                
                if success:
                    st.success(f"监测已启动！ID: {data['stream_id'][:8]}...")
                    st.rerun()
                else:
                    st.error(f"启动失败: {msg}")
    
    st.divider()
    
    # 显示活动视频流
    active_streams = load_active_streams()
    
    if not active_streams:
        st.info("暂无活动的视频流")
        return
    
    st.subheader(f"共有 {len(active_streams)} 个活动视频流")
    
    for stream in active_streams:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                **来源:** `{stream['source']}`  
                **ID:** `{stream['stream_id']}`  
                **启动时间:** {format_datetime(stream.get('started_at'))}  
                **过期时间:** {format_datetime(stream.get('expires_at'))}
                """)
            
            with col2:
                if st.button("观看", key=f"watch_{stream['stream_id']}"):
                    st.session_state.selected_stream = stream
            
            with col3:
                if st.button("停止", key=f"stop_{stream['stream_id']}", type="secondary"):
                    endpoint = API_ENDPOINTS['STREAMS_STOP'].format(stream['stream_id'])
                    success, _, msg = make_api_request('POST', endpoint)
                    if success:
                        st.success("视频流已停止")
                        st.rerun()
        
        # 显示选中的视频流
        if st.session_state.get('selected_stream') and st.session_state.selected_stream['stream_id'] == stream['stream_id']:
            st.image(
                stream['feed_url'],
                caption=f"实时视频流 - {stream['source']}",
                use_column_width=True
            )

# ==============================================================================
# 10. 检测记录页面
# ==============================================================================

def render_records_page():
    """检测记录页面"""
    st.title("📝 检测记录查询")
    
    if not st.session_state.api_status[0]:
        st.error("⚠️ API服务未连接")
        return
    
    # 筛选条件
    with st.expander("🔍 高级筛选", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name_filter = st.text_input("按姓名筛选", key="filter_name")
            sn_filter = st.text_input("按SN筛选", key="filter_sn")
        
        with col2:
            start_date = st.date_input("开始日期", key="filter_start")
            end_date = st.date_input("结束日期", key="filter_end")
        
        with col3:
            page_size = st.selectbox("每页显示", [10, 20, 50, 100], key="page_size")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("应用筛选", type="primary", use_container_width=True):
                st.session_state.detection_page = 1
                st.rerun()
        
        with col_btn3:
            if st.button("重置筛选", use_column_width=True):
                st.session_state.filters = {}
                st.rerun()
    
    # 构建筛选条件
    filters = {}
    if name_filter:
        filters['name'] = name_filter
    if sn_filter:
        filters['sn'] = sn_filter
    if start_date:
        filters['start_date'] = datetime.combine(start_date, datetime.min.time())
    if end_date:
        filters['end_date'] = datetime.combine(end_date, datetime.max.time())
    
    # 加载记录
    records_data = load_detection_records(
        page=st.session_state.detection_page,
        page_size=page_size,
        **filters
    )
    
    if not records_data:
        st.info("暂无检测记录")
        return
    
    # 显示统计信息
    total = records_data.get('total', 0)
    st.subheader(f"共找到 {total} 条记录")
    
    # 显示记录列表
    records = records_data.get('records', [])
    
    for record in records:
        with st.expander(f"{record.get('name')} - {format_datetime(record.get('detected_at'))}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image_url = format_image_url(record.get('image_url'))
                st.image(image_url, caption="检测图片", use_column_width=True)
            
            with col2:
                similarity = record.get('similarity', 0) * 100
                st.markdown(f"""
                **姓名:** {record.get('name')}  
                **SN:** {record.get('sn')}  
                **检测时间:** {format_datetime(record.get('detected_at'))}  
                **相似度:** <span class="success-text">{similarity:.1f}%</span>  
                **记录ID:** {record.get('id')}
                """)
    
    # 分页
    total_pages = records_data.get('total_pages', 1)
    current_page = records_data.get('page', 1)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        if st.button("上一页", disabled=current_page <= 1):
            st.session_state.detection_page = current_page - 1
            st.rerun()
        
        st.write(f"第 {current_page} 页 / 共 {total_pages} 页")
        
        if st.button("下一页", disabled=current_page >= total_pages):
            st.session_state.detection_page = current_page + 1
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 11. 统计分析页面
# ==============================================================================

def render_analytics_page():
    """统计分析页面"""
    st.title("📈 数据统计分析")
    
    if not st.session_state.api_status[0]:
        st.error("⚠️ API服务未连接")
        return
    
    # 加载所有图表数据
    load_charts_data()
    
    # 基本统计
    stats = load_detection_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("总检测次数", stats.get('total_detections', 0), "🔢"),
            ("检测人员数", stats.get('unique_persons', 0), "👥"),
            ("今日检测", stats.get('today_detections', 0), "📅"),
            ("本周检测", stats.get('week_detections', 0), "📊")
        ]
        
        for col, (label, value, icon) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # 图表区域
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 7天检测趋势")
        if st.session_state.get('weekly_trend'):
            data = st.session_state.weekly_trend
            df = pd.DataFrame({
                '日期': data.get('dates', []),
                '检测次数': data.get('counts', [])
            })
            
            chart = alt.Chart(df).mark_line(point=True, color='#667eea').encode(
                x=alt.X('日期', title='日期'),
                y=alt.Y('检测次数', title='检测次数'),
                tooltip=['日期', '检测次数']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("暂无趋势数据")
    
    with col2:
        st.subheader("🍩 人员检测分布")
        if st.session_state.get('person_pie_data'):
            data = st.session_state.person_pie_data
            df = pd.DataFrame({
                '人员': data.get('labels', []),
                '检测次数': data.get('values', [])
            })
            
            chart = alt.Chart(df).mark_arc().encode(
                theta='检测次数',
                color=alt.Color('人员', legend=None),
                tooltip=['人员', '检测次数']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("暂无分布数据")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("⏰ 检测时段分布")
        if st.session_state.get('hourly_trend'):
            data = st.session_state.hourly_trend
            df = pd.DataFrame({
                '小时': [f"{h:02d}:00" for h in data.get('hours', [])],
                '检测次数': data.get('counts', [])
            })
            
            chart = alt.Chart(df).mark_bar(color='#764ba2').encode(
                x='小时',
                y='检测次数',
                color=alt.Color('检测次数', scale=alt.Scale(scheme='viridis'))
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("暂无时段数据")
    
    with col4:
        st.subheader("🏆 检测排行榜")
        if st.session_state.get('top_persons'):
            data = st.session_state.top_persons
            persons = data.get('persons', [])
            
            for i, person in enumerate(persons[:10]):
                emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"][i]
                st.markdown(f"{emoji} **{person.get('name')}** - {person.get('count')} 次检测")

# ==============================================================================
# 12. 主应用
# ==============================================================================

def main():
    """主应用函数"""
    initialize_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 根据当前页面渲染内容
    page_map = {
        '仪表盘': render_dashboard_page,
        '人脸库管理': render_faces_management_page,
        '实时监测': render_monitoring_page,
        '检测记录': render_records_page,
        '统计分析': render_analytics_page
    }
    
    current_page = st.session_state.current_page
    if current_page in page_map:
        page_map[current_page]()
    else:
        render_dashboard_page()

if __name__ == "__main__":
    main()