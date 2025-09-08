# app/core/database/results_ops.py

"""
检测结果数据库操作模块
提供简洁、可靠的数据库操作接口
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError
from typing import List, Dict, Any, Optional
from app.cfg.logging import app_logger
from .models.detected_results import DetectedFace


def _validate_result_data(result: Dict[str, Any]) -> None:
    """
    验证单条检测结果数据
    
    Args:
        result: 检测结果数据
        
    Raises:
        ValueError: 数据验证失败
    """
    if not isinstance(result, dict):
        raise ValueError("检测结果必须是字典类型")
    
    # 检查必需字段
    required_fields = {'sn', 'name', 'task_id', 'app_id', 'app_name', 'domain_name'}
    missing_fields = required_fields - set(result.keys())
    if missing_fields:
        raise ValueError(f"缺少必需字段: {missing_fields}")
    
    # 基本数据验证
    if not result.get('sn') or not isinstance(result['sn'], str):
        raise ValueError("sn字段必须是非空字符串")
    
    if not result.get('name') or not isinstance(result['name'], str):
        raise ValueError("name字段必须是非空字符串")
    
    if not isinstance(result.get('task_id'), int):
        raise ValueError("task_id字段必须是整数")
    
    if not isinstance(result.get('app_id'), int):
        raise ValueError("app_id字段必须是整数")
    
    if not result.get('app_name') or not isinstance(result['app_name'], str):
        raise ValueError("app_name字段必须是非空字符串")
    
    if not result.get('domain_name') or not isinstance(result['domain_name'], str):
        raise ValueError("domain_name字段必须是非空字符串")
    
    # 验证相似度范围（如果存在）
    similarity = result.get('similarity')
    if similarity is not None and not (0.0 <= similarity <= 1.0):
        raise ValueError(f"相似度必须在0-1之间: {similarity}")


def insert_single_result(db: Session, result: Dict[str, Any]) -> int:
    """
    插入单条检测结果
    
    Args:
        db: 数据库会话
        result: 检测结果数据
        
    Returns:
        int: 插入记录的ID
        
    Raises:
        ValueError: 数据验证失败
        SQLAlchemyError: 数据库操作失败
    """
    try:
        # 数据验证
        _validate_result_data(result)
        
        # 创建数据库对象
        db_record = DetectedFace(**result)
        
        # 添加到会话并提交
        db.add(db_record)
        db.commit()
        
        # 刷新对象以获取ID
        db.refresh(db_record)
        
        app_logger.debug(f"成功插入单条检测结果，ID: {db_record.id}")
        return db_record.id
        
    except ValueError as e:
        app_logger.error(f"数据验证失败: {e}")
        raise
    except IntegrityError as e:
        db.rollback()
        app_logger.error(f"数据完整性错误: {e}")
        raise SQLAlchemyError(f"数据完整性错误: {e}")
    except DataError as e:
        db.rollback()
        app_logger.error(f"数据格式错误: {e}")
        raise SQLAlchemyError(f"数据格式错误: {e}")
    except SQLAlchemyError as e:
        db.rollback()
        app_logger.error(f"数据库操作失败: {e}")
        raise


def insert_batch_results(db: Session, results: List[Dict[str, Any]]) -> int:
    """
    批量插入检测结果到数据库
    
    Args:
        db: 数据库会话
        results: 检测结果列表
        
    Returns:
        int: 成功插入的记录数
        
    Raises:
        ValueError: 数据验证失败
        SQLAlchemyError: 数据库操作失败
    """
    if not results:
        app_logger.debug("批量插入：结果列表为空，跳过操作")
        return 0
    
    if not isinstance(results, list):
        raise ValueError("结果必须是列表类型")
    
    try:
        # 批量数据验证
        for i, result in enumerate(results):
            try:
                _validate_result_data(result)
            except ValueError as e:
                raise ValueError(f"第{i+1}条记录验证失败: {e}")
        
        # 使用bulk_insert_mappings提高性能
        db.bulk_insert_mappings(DetectedFace, results)
        db.commit()
        
        inserted_count = len(results)
        app_logger.info(f"成功批量插入 {inserted_count} 条检测结果")
        return inserted_count
        
    except ValueError as e:
        app_logger.error(f"批量插入数据验证失败: {e}")
        raise
    except IntegrityError as e:
        db.rollback()
        app_logger.error(f"批量插入数据完整性错误: {e}")
        raise SQLAlchemyError(f"数据完整性错误: {e}")
    except DataError as e:
        db.rollback()
        app_logger.error(f"批量插入数据格式错误: {e}")
        raise SQLAlchemyError(f"数据格式错误: {e}")
    except SQLAlchemyError as e:
        db.rollback()
        app_logger.error(f"批量数据库操作失败: {e}")
        raise


def delete_result_by_id(db: Session, result_id: int) -> bool:
    """
    根据ID删除检测结果
    
    Args:
        db: 数据库会话
        result_id: 结果ID
        
    Returns:
        bool: 删除是否成功
        
    Raises:
        SQLAlchemyError: 数据库操作失败
    """
    try:
        deleted_count = db.query(DetectedFace).filter(
            DetectedFace.id == result_id
        ).delete()
        
        db.commit()
        
        success = deleted_count > 0
        if success:
            app_logger.debug(f"成功删除检测结果，ID: {result_id}")
        else:
            app_logger.warning(f"删除失败，记录不存在，ID: {result_id}")
        
        return success
        
    except SQLAlchemyError as e:
        db.rollback()
        app_logger.error(f"删除操作失败: {e}")
        raise


def get_result_by_id(db: Session, result_id: int) -> Optional[DetectedFace]:
    """
    根据ID获取检测结果
    
    Args:
        db: 数据库会话
        result_id: 结果ID
        
    Returns:
        Optional[DetectedFace]: 检测结果对象，不存在则返回None
        
    Raises:
        SQLAlchemyError: 数据库操作失败
    """
    try:
        result = db.query(DetectedFace).filter(
            DetectedFace.id == result_id
        ).first()
        
        return result
        
    except SQLAlchemyError as e:
        app_logger.error(f"查询操作失败: {e}")
        raise
