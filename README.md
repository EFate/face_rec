project
├── app/  (核心应用)
│   ├── cfg/  (配置模块)
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── default.yaml
│   │   ├── development.yaml
│   │   ├── production.yaml
│   │   └── testing.yaml
│   ├── core/  (核心组件，如模型管理器)
│   │   └── model_manager.py
│   ├── router/  (API路由层)
│   │   └── face_router.py
│   ├── schema/  (数据模型/验证层)
│   │   └── face_schema.py
│   ├── service/  (业务逻辑层)
│   │   ├── face_service.py
│   │   ├── face_dao.py  (数据访问对象)
│   │   └── deepface_model/
│   ├── static/  (静态文件，如Swagger UI)
│   │   └── swagger-ui/
│   │       ├── swagger-ui-bundle.js
│   │       └── swagger-ui.css
│   ├── main.py  (FastAPI应用主入口)
│   └── __init__.py
│
├── data/  (数据存储)
│   ├── faces/  (人脸图片库)
│   │   └── sn001/
│   │       └── img_face_sn001_... .jpg
│   ├── face_features.csv  (人脸特征向量)
│   └── ds_model_... .pkl  (缓存或预处理的特征数据)
│
├── logs/  (日志文件)
├── run.py  (应用启动脚本)
实现一个高性能、易扩展、功能完善的人脸识别项目，核心需求包括:
1.完整的人脸库功能:增、删、改、查。
2.视频流预测与结果展示:实时识别视频流中的人脸，并将识别框和结果信息绘制在视频上返回
3.高性能与最佳实践:基于FastAPI和DeepFace，代码清晰、健壮、易于维护。
4.修复现有代码错误:解决现有代码中的逻辑缺陷和不合理之处。
5.中文回复，给出修改后的完整代码