# app/models/database.py

from sqlalchemy import  Column, Integer, String, Float, DateTime
from app.core.database.database import Base
from sqlalchemy.sql import func


class DetectedFace(Base):
    """
    人脸检测结果的数据表模型，仅包含核心信息和图片URL。
    """
    __tablename__ = 'detected_faces'

    # 主键，自增
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 记录创建时间，默认为当前时间
    create_time = Column(DateTime, server_default=func.now())

    # 记录更新时间，每次更新时自动刷新
    update_time = Column(DateTime, onupdate=func.now())

    # 识别出的sn
    sn = Column(String, nullable=False, index=True)
    # 识别出的姓名（例如：“Unknown”或实际姓名）
    name = Column(String)

    # 识别相似度分数
    similarity = Column(Float)

    # 本地图片的访问网址
    image_url = Column(String, nullable=False)

    def __repr__(self):
        return (f"<DetectedFace(id={self.id}, sn='{self.sn}', name='{self.name}', "
                f"similarity={self.similarity:.2f}, url='{self.image_url}')>")




