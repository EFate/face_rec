# app/service/face_dao.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from datetime import datetime
import uuid
import json

from sqlalchemy import create_engine, Column, String, DateTime, Text, update
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from app.cfg.logging import app_logger

# 定义抽象基类
class FaceDataDAO(ABC):
    @abstractmethod
    def create(self, name: str, sn: str, features: List[float], image_path: Path) -> Dict[str, Any]:
        """创建一个新的人脸特征记录。返回新记录的字典。"""
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有注册的人脸特征及其元数据。"""
        pass

    @abstractmethod
    def get_features_by_sn(self, sn: str) -> List[Dict[str, Any]]:
        """根据SN获取某个人的所有特征。"""
        pass

    @abstractmethod
    def delete_by_sn(self, sn: str) -> int:
        """根据SN删除所有相关人脸特征记录。返回删除的记录数量。"""
        pass

    @abstractmethod
    def update_by_sn(self, sn: str, update_data: Dict[str, Any]) -> int:
        """根据SN更新记录。返回更新的记录数量。"""
        pass

# ... CSVFaceDataDAO (保持不变，但仍不推荐在生产中使用) ...
class CSVFaceDataDAO(FaceDataDAO):
    def __init__(self, features_path: Path):
        self.features_path = features_path
        self.feature_cols = [f"feature_{i}" for i in range(512)]
        self.columns = ["uuid", "name", "sn", "registration_time", "image_path"] + self.feature_cols
        self._ensure_csv_file()

    def _ensure_csv_file(self):
        if not self.features_path.exists() or self.features_path.stat().st_size == 0:
            pd.DataFrame(columns=self.columns).to_csv(str(self.features_path), index=False)
            
    def _read_df(self) -> pd.DataFrame:
        try:
            return pd.read_csv(str(self.features_path))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame(columns=self.columns)

    def create(self, name: str, sn: str, features: List[float], image_path: Path) -> Dict[str, Any]:
        df = self._read_df()
        new_uuid = str(uuid.uuid4())
        registration_time = datetime.now()
        new_row_data = {
            "uuid": new_uuid, "name": name, "sn": sn,
            "registration_time": registration_time.isoformat(),
            "image_path": str(image_path),
            **{col: f for col, f in zip(self.feature_cols, features)}
        }
        new_row_df = pd.DataFrame([new_row_data])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(str(self.features_path), index=False)
        return {
            "uuid": new_uuid, "name": name, "sn": sn, 
            "registration_time": registration_time, 
            "image_path": str(image_path), "features": features
        }

    def get_all(self) -> List[Dict[str, Any]]:
        df = self._read_df()
        if df.empty: return []
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            record['features'] = [row[col] for col in self.feature_cols]
            for col in self.feature_cols:
                del record[col]
            records.append(record)
        return records

    def get_features_by_sn(self, sn: str) -> List[Dict[str, Any]]:
        df = self._read_df()
        df_filtered = df[df["sn"] == sn]
        if df_filtered.empty: return []
        
        results = []
        for _, row in df_filtered.iterrows():
            record = row.to_dict()
            record['features'] = [row[col] for col in self.feature_cols]
            for col in self.feature_cols:
                del record[col]
            results.append(record)
        return results

    def delete_by_sn(self, sn: str) -> int:
        df = self._read_df()
        if df.empty: return 0
        initial_rows = len(df)
        df_filtered = df[df["sn"] != sn]
        if len(df_filtered) == initial_rows: return 0
        df_filtered.to_csv(str(self.features_path), index=False)
        return initial_rows - len(df_filtered)

    def update_by_sn(self, sn: str, update_data: Dict[str, Any]) -> int:
        app_logger.warning("CSV后端性能低下，更新操作会重写整个文件，不建议在生产中使用。")
        df = self._read_df()
        if df.empty or sn not in df['sn'].values:
            return 0
        
        update_indices = df['sn'] == sn
        updated_count = update_indices.sum()
        for key, value in update_data.items():
            if key in df.columns:
                df.loc[update_indices, key] = value
        df.to_csv(str(self.features_path), index=False)
        return int(updated_count)

# SQLAlchemy Model for SQLite
Base = declarative_base()
class FaceFeature(Base):
    __tablename__ = 'face_features'
    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    sn = Column(String, nullable=False, index=True)
    registration_time = Column(DateTime, default=datetime.now)
    image_path = Column(String, nullable=False)
    features_json = Column(Text, nullable=False)

# SQLite 实现
class SQLiteFaceDataDAO(FaceDataDAO):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def _to_dict(self, f: FaceFeature) -> Dict[str, Any]:
        return {
            "uuid": f.uuid, "name": f.name, "sn": f.sn,
            "registration_time": f.registration_time,
            "image_path": f.image_path,
            "features": json.loads(f.features_json)
        }

    def create(self, name: str, sn: str, features: List[float], image_path: Path) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            new_feature = FaceFeature(
                name=name, sn=sn,
                features_json=json.dumps(features),
                image_path=str(image_path)
            )
            session.add(new_feature)
            session.commit()
            session.refresh(new_feature)
            return self._to_dict(new_feature)
        except SQLAlchemyError as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"数据库错误: {e}")
        finally:
            session.close()

    def get_all(self) -> List[Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            all_features = session.query(FaceFeature).all()
            return [self._to_dict(f) for f in all_features]
        finally:
            session.close()

    def get_features_by_sn(self, sn: str) -> List[Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            features = session.query(FaceFeature).filter(FaceFeature.sn == sn).all()
            return [self._to_dict(f) for f in features]
        finally:
            session.close()
            
    def delete_by_sn(self, sn: str) -> int:
        session = self.SessionLocal()
        try:
            deleted_count = session.query(FaceFeature).filter(FaceFeature.sn == sn).delete(synchronize_session=False)
            session.commit()
            return deleted_count
        except SQLAlchemyError as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"数据库错误: {e}")
        finally:
            session.close()

    def update_by_sn(self, sn: str, update_data: Dict[str, Any]) -> int:
        session = self.SessionLocal()
        if not update_data:
            return 0
        try:
            stmt = (
                update(FaceFeature)
                .where(FaceFeature.sn == sn)
                .values(**update_data)
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
        except SQLAlchemyError as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"数据库错误: {e}")
        finally:
            session.close()