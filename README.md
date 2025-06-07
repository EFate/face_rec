# 核心依赖

"fastapi[standard]"
pydantic
pydantic-settings
typer
python-dotenv
loguru
pyyaml
sqlalchemy
aiofiles

# 人脸识别 (InsightFace)

insightface
onnx

# 根据您的硬件选择一个

# onnxruntime # CPU 版本

# onnxruntime-gpu # GPU 版本

# 其他

opencv-python
numpy
pandas
scipy

project
├── app/ (核心应用)
│ ├── cfg/ (配置模块)
│ │ ├── config.py
│ │ ├── logging.py
│ │ ├── default.yaml
│ │ └── ...
│ ├── core/ (核心组件)
│ │ └── model*manager.py
│ ├── router/ (API 路由层)
│ │ └── face_router.py
│ ├── schema/ (数据模型/验证层)
│ │ └── face_schema.py
│ ├── service/ (业务逻辑层)
│ │ ├── face_service.py
│ │ └── face_dao.py
│ ├── static/ (静态文件)
│ │ └── swagger-ui/
│ ├── main.py (FastAPI 应用主入口)
│ └── **init**.py
│
├── data/ (所有数据存储)
│ ├── .insightface/ <- [新增] InsightFace 模型将下载并存储于此
│ │ └── models/
│ │ └── buffalo_l/
│ │ ├── det_10g.onnx
│ │ ├── genderage.onnx
│ │ └── w600k_r50.onnx
│ ├── faces/ (注册的人脸图片库)
│ │ └── sn001/
│ │ └── face_sn001*... .jpg
│ └── face_features.db (SQLite 数据库文件)
│
├── logs/ (日志文件)
├── run.py (应用启动脚本)
└── requirements.txt (项目依赖)
实现一个高性能、易扩展、功能完善的人脸识别项目，核心需求包括: 1.完整的人脸库功能:增、删、改、查。 2.视频流预测与结果展示:实时识别视频流中的人脸，并将识别框和结果信息绘制在视频上返回 3.高性能与最佳实践:基于 FastAPI 和 InsightFace，代码清晰、健壮、易于维护。 4.参考现有架构和代码，去除所有 deepface 逻辑，将其改为 InsightFace 逻辑;
5、修改 face_schema.py 让其符合 InsightFace，重新思考所有代码逻辑，让这个项目符合最佳实践。
5、修改 config.py 的 deepface 逻辑，将其改为 InsightFace 逻辑。 7.中文回复，给出修改后的完整代码
