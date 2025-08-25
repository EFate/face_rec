# ui.py
import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any, Dict, List, Optional
import os
import json
from datetime import datetime

# ==============================================================================
# 1. 页面配置与样式 (Page Config & Styling)
# ==============================================================================
st.set_page_config(
    page_title="人脸识别智能管理系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 现代化的CSS样式 ---
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
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }

    /* --- 指标卡片 --- */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .metric-card .title { font-weight: 600; color: #6c757d; font-size: 1rem; margin-bottom: 10px; }
    .metric-card .value { font-weight: 700; color: #1a1f36; font-size: 2.2rem; }
    .metric-card .icon { font-size: 2.5rem; text-align: right; opacity: 0.8; }
    .metric-card.ok { border-left: 5px solid #28a745; }
    .metric-card.error { border-left: 5px solid #dc3545; }

    /* --- 其他美化 --- */
    .stButton>button { border-radius: 8px; font-weight: 600; }
    [data-testid="stExpander"] { border-radius: 8px; }
    [data-testid="stFileUploader"] { padding: 10px; background-color: #fafafa; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. API客户端 (API Client)
# ==============================================================================
class ApiClient:
    """一个用于与后端API交互的客户端类"""

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
        """统一的内部请求方法"""
        url = f"{self.base_url}{kwargs.pop('url_format', self.endpoints[endpoint_key])}"
        try:
            response = requests.request(method, url, timeout=15, **kwargs)
            if response.ok:
                if response.status_code == 204 or not response.content:
                    return True, {"msg": "操作成功"}
                res_json = response.json()
                if res_json.get("code") == 0:
                    return True, res_json.get("data", {})
                return False, res_json.get("msg", "后端返回业务错误")
            else:
                try:
                    # 尝试解析FastAPI的错误详情
                    detail = response.json().get("detail", "未知错误")
                    if isinstance(detail, list):
                        detail = detail[0].get('msg', '请求验证失败')
                    return False, f"HTTP {response.status_code}: {detail}"
                except json.JSONDecodeError:
                    return False, f"HTTP {response.status_code}: 无法解析响应"
        except requests.RequestException as e:
            return False, f"网络请求失败: {e}"

    # --- Wrapper methods for each endpoint ---
    def check_health(self): return self._request('GET', 'health')
    def get_all_faces(self): return self._request('GET', 'faces')
    def register_face(self, data, files): return self._request('POST', 'faces', data=data, files=files)
    def update_face(self, sn, name): return self._request('PUT', 'face_by_sn', url_format=self.endpoints['face_by_sn'].format(sn), json={"name": name})
    def delete_face(self, sn): return self._request('DELETE', 'face_by_sn', url_format=self.endpoints['face_by_sn'].format(sn))
    def recognize_face(self, files): return self._request('POST', 'recognize', files=files)
    def start_stream(self, source, lifetime): return self._request('POST', 'streams_start', json={"source": source, "lifetime_minutes": lifetime})
    def stop_stream(self, stream_id): return self._request('POST', 'streams_stop', url_format=self.endpoints['streams_stop'].format(stream_id))
    def list_streams(self): return self._request('GET', 'streams_list')
    def get_detection_stats(self): return self._request('GET', 'stats')
    def get_weekly_trend(self): return self._request('GET', 'weekly_trend')
    def get_detection_records(self, params): return self._request('GET', 'records', params=params)
    def get_person_pie_data(self): return self._request('GET', 'person_pie')
    def get_hourly_trend_data(self): return self._request('GET', 'hourly_trend')
    def get_top_persons_data(self, limit=10): return self._request('GET', 'top_persons', params={'limit': limit})


# ==============================================================================
# 3. 会话状态管理 (Session State)
# ==============================================================================
def initialize_session_state():
    """初始化应用所需的全部会话状态。"""
    if "app_state" not in st.session_state:
        backend_host = os.getenv("HOST__IP", "localhost")
        backend_port = os.getenv("SERVER__PORT", "12010")
        st.session_state.app_state = {
            "api_url": f"{backend_host}:{backend_port}",
            "api_client": ApiClient(f"{backend_host}:{backend_port}"),
            "api_status": (False, "尚未连接"),
            "active_page": "数据看板",
            "faces_data": {"count": 0, "faces": [], "unique_sns": []},
            "management": {
                "show_register_dialog": False,
                "selected_sn": None
            },
            "monitoring": {
                "viewing_stream_info": None
            },
            "analytics": {
                "records_page": 1,
                "records_filters": {"name": "", "sn": "", "start_date": None, "end_date": None}
            }
        }


# ==============================================================================
# 4. UI 渲染模块 (UI Rendering Modules)
# ==============================================================================

def render_sidebar():
    with st.sidebar:
        st.title("🤖 人脸识别系统")
        st.caption("v2.0 - 智能管理版")

        # --- API配置 ---
        api_url = st.text_input("后端服务地址", value=st.session_state.app_state['api_url'])
        if api_url != st.session_state.app_state['api_url']:
            st.session_state.app_state['api_url'] = api_url
            st.session_state.app_state['api_client'] = ApiClient(api_url)
            st.rerun()

        client = st.session_state.app_state['api_client']
        success, data = client.check_health()
        status_msg = data.get('message', "连接失败") if success else data
        st.session_state.app_state['api_status'] = (success, status_msg)
        status_icon = "✅" if success else "❌"
        st.info(f"**API状态:** {status_msg}", icon=status_icon)
        st.divider()

        # --- 导航 ---
        pages = ["数据看板", "人脸库管理", "实时监测", "检测分析"]
        st.session_state.app_state['active_page'] = st.radio("导航", pages, label_visibility="collapsed")
        
        st.divider()
        if st.button("🔄 强制刷新全站数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

def render_dashboard_page():
    st.header("📊 数据看板总览")
    client = st.session_state.app_state['api_client']

    # --- 获取核心数据 ---
    @st.cache_data(ttl=30)
    def get_dashboard_data():
        stats_success, stats_data = client.get_detection_stats()
        faces_success, faces_data = client.get_all_faces()
        streams_success, streams_data = client.list_streams()
        trend_success, trend_data = client.get_weekly_trend()
        return {
            "stats": stats_data if stats_success else {},
            "faces": faces_data if faces_success else {},
            "streams": streams_data if streams_success else {},
            "trend": trend_data if trend_success else {}
        }

    data = get_dashboard_data()

    # --- 指标卡片 ---
    api_status, api_color_class = ("在线", "ok") if st.session_state.app_state['api_status'][0] else ("离线", "error")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.html(f"""<div class="metric-card"><div class="title">人脸库人员总数</div><div class="value">{data.get('faces', {}).get('count', 'N/A')}</div></div>""")
    with col2:
        st.html(f"""<div class="metric-card"><div class="title">总检测次数</div><div class="value">{data.get('stats', {}).get('total_detections', 'N/A')}</div></div>""")
    with col3:
        st.html(f"""<div class="metric-card"><div class="title">今日检测</div><div class="value">{data.get('stats', {}).get('today_detections', 'N/A')}</div></div>""")
    with col4:
        st.html(f"""<div class="metric-card {api_color_class}"><div class="title">API 服务</div><div class="value">{api_status}</div></div>""")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 图表与最新记录 ---
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        with st.container(border=True):
            st.subheader("🗓️ 近7日检测趋势")
            trend_df = pd.DataFrame(data.get('trend', {}).get('trend_data', []))
            if not trend_df.empty:
                trend_df['date'] = pd.to_datetime(trend_df['date'])
                st.line_chart(trend_df, x='date', y='count')
            else:
                st.info("暂无趋势数据。")
    with col2:
        with st.container(border=True, height=380):
            st.subheader("⏱️ 最新检测记录")
            recent = data.get('stats', {}).get('recent_detections', [])
            if recent:
                for item in recent:
                    col_img, col_info = st.columns([0.2, 0.8])
                    col_img.image(item['image_url'], width=50)
                    col_info.markdown(f"**{item['name']}** ({item['sn']})")
                    col_info.caption(f"{datetime.fromisoformat(item['create_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("暂无最近检测记录。")

    st.divider()
    # --- 快速识别工具 ---
    with st.expander("🧐 快速人脸识别", expanded=True):
        uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "png", "jpeg"], key="recognize_uploader")
        if uploaded_file:
            img_col, res_col = st.columns(2)
            img_col.image(uploaded_file, caption="待识别图片")
            with res_col:
                with st.spinner("正在识别..."):
                    files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                    success, results = client.recognize_face(files)
                if success:
                    if results:
                        st.success(f"识别成功！找到 {len(results)} 个匹配项。")
                        for res in results:
                            st.info(f"**姓名:** {res['name']} | **SN:** {res['sn']}\n**相似度:** {res['similarity']:.2%}")
                    else:
                        st.info("检测到人脸，但未在库中找到匹配项。")
                else:
                    st.error(f"识别失败: {results}")

def render_management_page():
    st.header("🗂️ 人脸库管理中心")
    client = st.session_state.app_state['api_client']

    # --- 注册弹窗 ---
    @st.dialog("➕ 注册新人员", width="large")
    def register_dialog():
        with st.form("register_form"):
            name = st.text_input("姓名", placeholder="例如：张三")
            sn = st.text_input("唯一编号(SN)", placeholder="例如：EMP001")
            image_file = st.file_uploader("上传人脸照片", type=["jpg", "png", "jpeg"])
            submitted = st.form_submit_button("✔️ 确认注册", type="primary", use_container_width=True)
            if submitted:
                if not all([name, sn, image_file]):
                    st.warning("所有字段均为必填项。")
                    return
                with st.spinner("注册中..."):
                    success, msg = client.register_face(
                        data={'name': name, 'sn': sn},
                        files={'image_file': (image_file.name, image_file.getvalue())}
                    )
                if success:
                    st.toast("注册成功！", icon="🎉")
                    st.cache_data.clear()
                    st.session_state.app_state['management']['show_register_dialog'] = False
                    st.rerun()
                else:
                    st.error(f"注册失败: {msg}")

    if st.button("➕ 注册新人员", type="primary"):
        st.session_state.app_state['management']['show_register_dialog'] = True

    if st.session_state.app_state['management']['show_register_dialog']:
        register_dialog()

    st.divider()

    # --- 人员列表与管理 ---
    @st.cache_data(ttl=60)
    def get_faces_data():
        success, data = client.get_all_faces()
        return pd.DataFrame(data.get('faces', [])) if success else pd.DataFrame()

    faces_df = get_faces_data()
    if faces_df.empty:
        st.info("人脸库为空或加载失败，请尝试刷新。")
        return

    # 按SN分组
    persons_df = faces_df.groupby('sn').agg(
        name=('name', 'first'),
        registrations=('uuid', 'count')
    ).reset_index()

    st.subheader(f"👥 人员列表 (共 {len(persons_df)} 人)")
    
    col_table, col_detail = st.columns([0.5, 0.5])

    with col_table:
        selected = st.radio(
            "选择人员进行管理:",
            options=persons_df['sn'],
            format_func=lambda sn: f"{persons_df[persons_df['sn'] == sn]['name'].values[0]} ({sn})",
            label_visibility="collapsed"
        )
        st.session_state.app_state['management']['selected_sn'] = selected

    with col_detail, st.container(border=True):
        sn = st.session_state.app_state['management']['selected_sn']
        if sn:
            person_details = faces_df[faces_df['sn'] == sn]
            name = person_details.iloc[0]['name']
            
            st.subheader(f"👤 {name} (SN: {sn})")
            
            # 显示所有注册照片
            st.write("**已注册照片:**")
            img_urls = [row['image_url'] for _, row in person_details.iterrows()]
            st.image(img_urls, width=80)
            
            st.divider()

            with st.expander("⚙️ 管理选项"):
                # 更新
                new_name = st.text_input("更新姓名", value=name, key=f"update_{sn}")
                if st.button("✔️ 确认更新", key=f"update_btn_{sn}", use_container_width=True):
                    if new_name and new_name != name:
                        with st.spinner("更新中..."):
                            success, msg = client.update_face(sn, new_name)
                        if success:
                            st.toast(f"'{name}' 已更新为 '{new_name}'", icon="✅")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"更新失败: {msg}")

                # 删除
                if st.button("🗑️ 删除此人所有记录", type="secondary", use_container_width=True, key=f"delete_{sn}"):
                    with st.spinner("删除中..."):
                        success, msg = client.delete_face(sn)
                    if success:
                        st.toast(f"'{name}' ({sn}) 已被删除。", icon="🗑️")
                        st.cache_data.clear()
                        st.session_state.app_state['management']['selected_sn'] = None
                        st.rerun()
                    else:
                        st.error(f"删除失败: {msg}")

def render_monitoring_page():
    st.header("🛰️ 实时视频监测")
    client = st.session_state.app_state['api_client']

    with st.expander("▶️ 启动新监测任务", expanded=True):
        with st.form("start_stream_form"):
            source = st.text_input("视频源", "0", help="摄像头ID(0, 1) 或 视频文件/URL")
            lifetime = st.number_input("生命周期(分钟)", min_value=-1, value=10, help="-1 代表永久")
            if st.form_submit_button("🚀 开启监测", use_container_width=True, type="primary"):
                with st.spinner("请求启动视频流..."):
                    success, data = client.start_stream(source, lifetime)
                if success:
                    st.toast(f"视频流任务已启动！", icon="🚀")
                    st.session_state.app_state['monitoring']['viewing_stream_info'] = data
                    st.rerun()
                else:
                    st.error(f"启动失败: {data}")

    # 显示当前观看的视频流
    stream_info = st.session_state.app_state['monitoring'].get('viewing_stream_info')
    if stream_info:
        with st.container(border=True):
            st.subheader(f"正在播放: `{stream_info['source']}`")
            st.caption(f"Stream ID: `{stream_info['stream_id']}`")
            st.image(stream_info['feed_url'])
    else:
        st.info("请从下方列表选择一个流进行观看，或启动一个新任务。")
    st.divider()

    # 显示活动视频流列表
    st.subheader("所有活动中的监测任务")
    @st.cache_data(ttl=5)
    def get_active_streams():
        success, data = client.list_streams()
        return data.get('streams', []) if success else []

    active_streams = get_active_streams()
    if not active_streams:
        st.info("目前没有正在运行的视频监测任务。")
    else:
        for stream in active_streams:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    expires_at_str = stream.get('expires_at')
                    expires_display = datetime.fromisoformat(expires_at_str).strftime('%H:%M:%S') if expires_at_str else "永久"
                    st.markdown(f"**来源:** `{stream['source']}` | **过期时间:** {expires_display}")
                    st.caption(f"ID: `{stream['stream_id']}`")
                with col2:
                    btn_cols = st.columns(2)
                    if btn_cols[0].button("👁️ 观看", key=f"view_{stream['stream_id']}", use_container_width=True):
                        st.session_state.app_state['monitoring']['viewing_stream_info'] = stream
                        st.rerun()
                    if btn_cols[1].button("⏹️ 停止", key=f"stop_{stream['stream_id']}", use_container_width=True):
                        with st.spinner("停止中..."):
                            success, _ = client.stop_stream(stream['stream_id'])
                        if success:
                            st.toast("视频流已停止。", icon="✅")
                            if stream_info and stream_info['stream_id'] == stream['stream_id']:
                                st.session_state.app_state['monitoring']['viewing_stream_info'] = None
                            st.rerun()

def render_analytics_page():
    st.header("🔍 检测分析中心")
    client = st.session_state.app_state['api_client']

    tab1, tab2 = st.tabs(["📊 统计图表", "🗂️ 历史记录"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1, st.container(border=True, height=450):
            st.subheader("👥 人员检测分布")
            success, data = client.get_person_pie_data()
            if success and data.get('pie_data'):
                pie_df = pd.DataFrame(data['pie_data'])
                # 为了美观，将占比小的合并为 "其他"
                pie_df.loc[pie_df['percentage'] < 2, 'name'] = '其他'
                pie_df = pie_df.groupby('name')['count'].sum().reset_index()
                st.vega_lite_chart(pie_df, {
                    'mark': {'type': 'arc', 'innerRadius': 50},
                    'encoding': {
                        'theta': {'field': 'count', 'type': 'quantitative'},
                        'color': {'field': 'name', 'type': 'nominal', 'title': '姓名'},
                    },
                }, use_container_width=True)
            else:
                st.info("暂无饼图数据。")
        
        with col2, st.container(border=True, height=450):
            st.subheader("🏆 检测次数排行榜 (Top 10)")
            success, data = client.get_top_persons_data(limit=10)
            if success and data.get('top_persons'):
                top_df = pd.DataFrame(data['top_persons'])
                st.dataframe(top_df[['rank', 'name', 'sn', 'count']], hide_index=True, use_container_width=True)
            else:
                st.info("暂无排行数据。")

        with st.container(border=True):
            st.subheader("🕒 24小时检测活跃度")
            success, data = client.get_hourly_trend_data()
            if success and data.get('hourly_data'):
                hourly_df = pd.DataFrame(data['hourly_data'])
                st.bar_chart(hourly_df, x='hour', y='count')
            else:
                st.info("暂无小时趋势数据。")

    with tab2:
        st.subheader("历史检测记录查询")
        # --- 筛选器 ---
        with st.form("filter_form"):
            cols = st.columns(4)
            name = cols[0].text_input("按姓名筛选")
            sn = cols[1].text_input("按SN筛选")
            start_date = cols[2].date_input("开始日期", value=None)
            end_date = cols[3].date_input("结束日期", value=None)
            submitted = st.form_submit_button("🔍 查询")

        # --- 数据查询与展示 ---
        params = {
            "page": st.session_state.app_state['analytics'].get('records_page', 1),
            "page_size": 10,
            "name": name if name else None,
            "sn": sn if sn else None,
            "start_date": start_date.strftime('%Y-%m-%dT00:00:00') if start_date else None,
            "end_date": end_date.strftime('%Y-%m-%dT23:59:59') if end_date else None,
        }
        
        @st.cache_data(ttl=10)
        def get_records(p):
            return client.get_detection_records(params={k: v for k, v in p.items() if v is not None})
        
        success, data = get_records(params)

        if success and data.get('records'):
            df = pd.DataFrame(data['records'])
            df['detected_at'] = pd.to_datetime(df['create_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # --- 使用st.column_config美化表格 ---
            st.dataframe(
                df,
                column_config={
                    "image_url": st.column_config.ImageColumn("抓拍图", width="small"),
                    "name": "姓名",
                    "sn": "SN",
                    "similarity": st.column_config.ProgressColumn("相似度", format="%.2f", min_value=0, max_value=1),
                    "detected_at": "检测时间",
                },
                column_order=("image_url", "name", "sn", "similarity", "detected_at"),
                hide_index=True,
                use_container_width=True
            )
            # --- 分页 ---
            total_pages = data.get('total_pages', 1)
            page_cols = st.columns([0.8, 0.2])
            page_cols[0].write(f"总计 {data.get('total')} 条记录，共 {total_pages} 页")
            page_cols[1].number_input("页码", min_value=1, max_value=total_pages, key="analytics_records_page")

        elif success:
            st.info("在当前筛选条件下未找到任何记录。")
        else:
            st.error(f"加载记录失败: {data}")

# ==============================================================================
# 5. 主程序入口 (Main Application)
# ==============================================================================
def main():
    initialize_session_state()
    
    if not st.session_state.app_state['api_status'][0]:
        st.warning("API服务未连接，请在左侧侧边栏配置正确的服务地址并确保后端服务已启动。页面功能将受限。")

    render_sidebar()

    page_map = {
        "数据看板": render_dashboard_page,
        "人脸库管理": render_management_page,
        "实时监测": render_monitoring_page,
        "检测分析": render_analytics_page,
    }
    
    active_page_func = page_map.get(st.session_state.app_state['active_page'])
    if active_page_func:
        active_page_func()

if __name__ == "__main__":
    main()