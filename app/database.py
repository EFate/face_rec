# app/database.py
# 导入必要的模块
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# 从配置文件导入数据库文件路径
from app.cfg.config import SQLLITE_FILE

# 构造数据库URL
DATABASE_URL = f"sqlite:///{SQLLITE_FILE.resolve()}"
SQLLITE_FILE.parent.mkdir(parents=True, exist_ok=True)

# 创建数据库引擎
# 为SQLite添加 connect_args={"check_same_thread": False} 以支持多线程访问
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建ORM模型的基类
Base = declarative_base()

# 获取数据库会话的依赖函数
def get_db_session():
    """
    提供一个数据库会话的生成器，并确保在使用后关闭。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 7. 定义一个函数来创建表，确保应用启动时表结构存在
def init_db():
    """
    导入所有相关的模型并创建对应的数据库表。
    在函数内部导入模型可以避免循环依赖问题。
    """
    from app.models.detected_results import DetectedFace
    Base.metadata.create_all(bind=engine)