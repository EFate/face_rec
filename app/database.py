# 导入必要的模块
import os
from sqlalchemy import create_engine  # 导入创建数据库引擎的函数
from sqlalchemy.orm import sessionmaker  # 导入会话工厂
from sqlalchemy.orm import declarative_base, Session  # 导入基类和会话类型


# 创建数据库引擎，使用环境变量中的数据库URL
engine = create_engine(os.getenv("DATABASE_URL"))

# 创建会话工厂，绑定到数据库引擎
SessionLocal = sessionmaker(bind=engine)


# 创建ORM模型的基类
Base = declarative_base()

# 创建所有在 Base 中定义的表
Base.metadata.create_all(engine)

# 获取数据库会话的依赖函数
def get_db_session():
    db = SessionLocal()  # 创建新的数据库会话
    try:
        yield db  # 返回会话供使用
    finally:
        db.close()  # 确保会话最终被关闭
