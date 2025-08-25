# app/core/database/results_ops.py

# 导入SQLAlchemy会话
from sqlalchemy.orm import Session

# 导入结果模型
from .models.detected_results import DetectedFace

def insert_new_result(
    db: Session,
    new_result
) -> DetectedFace:
    new_result = DetectedFace(**new_result)

    db.add(new_result)

    db.flush()

    db.refresh(new_result)

    db.commit()

    return new_result
