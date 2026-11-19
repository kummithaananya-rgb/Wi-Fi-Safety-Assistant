"""
Database models and connection for Wi-Fi Safety Assistant.
Supports MySQL (production) with SQLite fallback for development.
"""
import uuid
import datetime as dt
import logging
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, Integer,
    Float, Boolean, JSON, Enum as SAEnum
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    """Stores every agent workflow task."""
    __tablename__ = "task_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc),
                        onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Input
    wifi_name = Column(String(255))
    connection_type = Column(String(100))
    location = Column(String(255))
    user_context = Column(Text)

    # Workflow
    status = Column(SAEnum("pending", "running", "complete", "error",
                           name="task_status"), default="pending")
    risk_level = Column(SAEnum("LOW", "MEDIUM", "HIGH", "UNKNOWN",
                               name="risk_level"), default="UNKNOWN")

    # Agent outputs (JSON blobs)
    agent1_output = Column(JSON)
    agent2_output = Column(JSON)
    agent3_output = Column(JSON)
    supervisor_output = Column(JSON)

    # Metrics
    total_duration_ms = Column(Float)
    agent1_duration_ms = Column(Float)
    agent2_duration_ms = Column(Float)
    agent3_duration_ms = Column(Float)
    models_used = Column(JSON)      # list of model names used
    guardrail_flags = Column(JSON)  # any guardrail interventions
    error_message = Column(Text)


class ModelCallLog(Base):
    """Tracks every AI model API call for performance analytics."""
    __tablename__ = "model_call_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36))
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    provider = Column(String(50))   # provider used at call time; historical names retained
    model_id = Column(String(100))
    agent_name = Column(String(50))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    duration_ms = Column(Float)
    success = Column(Boolean, default=True)
    error = Column(Text)


class DatabaseManager:
    _instance = None

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.engine = create_engine(
                self.db_url,
                pool_pre_ping=True,
                connect_args={"check_same_thread": False}
                if "sqlite" in self.db_url else {},
            )
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Database connected: {self.db_url.split('://')[0]}")
        except Exception as e:
            logger.error(f"Database init failed: {e}")
            # Fallback to SQLite
            if "sqlite" not in self.db_url:
                logger.warning("Falling back to SQLite")
                self.db_url = "sqlite:///wifi_safety.db"
                self._init_engine()

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ---- Task helpers ----

    def create_task(self, wifi_name: str, connection_type: str,
                    location: str, user_context: str) -> str:
        task_id = str(uuid.uuid4())
        with self.get_session() as s:
            task = TaskRecord(
                id=task_id,
                wifi_name=wifi_name,
                connection_type=connection_type,
                location=location,
                user_context=user_context,
                status="pending",
            )
            s.add(task)
            s.commit()
        return task_id

    def update_task(self, task_id: str, **kwargs):
        with self.get_session() as s:
            task = s.get(TaskRecord, task_id)
            if task:
                for k, v in kwargs.items():
                    setattr(task, k, v)
                task.updated_at = dt.datetime.now(dt.timezone.utc)
                s.commit()

    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.get_session() as s:
            task = s.get(TaskRecord, task_id)
            if task:
                return {
                    "id": task.id,
                    "created_at": task.created_at.isoformat(),
                    "wifi_name": task.wifi_name,
                    "connection_type": task.connection_type,
                    "location": task.location,
                    "status": task.status,
                    "risk_level": task.risk_level,
                    "agent1_output": task.agent1_output,
                    "agent2_output": task.agent2_output,
                    "agent3_output": task.agent3_output,
                    "supervisor_output": task.supervisor_output,
                    "total_duration_ms": task.total_duration_ms,
                    "models_used": task.models_used,
                    "guardrail_flags": task.guardrail_flags,
                    "error_message": task.error_message,
                }
        return None

    def get_recent_tasks(self, limit: int = 20) -> List[Dict]:
        with self.get_session() as s:
            tasks = (
                s.query(TaskRecord)
                .order_by(TaskRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": t.id,
                    "created_at": t.created_at.isoformat(),
                    "wifi_name": t.wifi_name,
                    "risk_level": t.risk_level,
                    "status": t.status,
                    "total_duration_ms": t.total_duration_ms,
                }
                for t in tasks
            ]

    def log_model_call(self, task_id: str, provider: str, model_id: str,
                       agent_name: str, duration_ms: float, success: bool,
                       prompt_tokens: int = 0, completion_tokens: int = 0,
                       error: str = None):
        with self.get_session() as s:
            log = ModelCallLog(
                task_id=task_id,
                provider=provider,
                model_id=model_id,
                agent_name=agent_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                success=success,
                error=error,
            )
            s.add(log)
            s.commit()

    def get_analytics(self) -> Dict[str, Any]:
        """Return aggregate performance analytics."""
        with self.get_session() as s:
            total_tasks = s.query(TaskRecord).count()
            complete = s.query(TaskRecord).filter_by(status="complete").count()
            errors = s.query(TaskRecord).filter_by(status="error").count()

            risk_counts = {}
            for lvl in ("LOW", "MEDIUM", "HIGH"):
                risk_counts[lvl] = (
                    s.query(TaskRecord).filter_by(risk_level=lvl).count()
                )

            calls = s.query(ModelCallLog).all()
            provider_stats: Dict[str, Any] = {}
            for c in calls:
                p = c.provider
                if p not in provider_stats:
                    provider_stats[p] = {"total": 0, "success": 0, "total_ms": 0.0}
                provider_stats[p]["total"] += 1
                if c.success:
                    provider_stats[p]["success"] += 1
                provider_stats[p]["total_ms"] += (c.duration_ms or 0)

            # Compute averages
            for p, stats in provider_stats.items():
                stats["success_rate"] = (
                    round(stats["success"] / stats["total"] * 100, 1)
                    if stats["total"] > 0 else 0
                )
                stats["avg_ms"] = (
                    round(stats["total_ms"] / stats["total"], 1)
                    if stats["total"] > 0 else 0
                )

            return {
                "total_tasks": total_tasks,
                "complete": complete,
                "errors": errors,
                "completion_rate": round(complete / total_tasks * 100, 1)
                if total_tasks > 0 else 0,
                "risk_distribution": risk_counts,
                "provider_stats": provider_stats,
            }


# Singleton
_db: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        from config import Config
        _db = DatabaseManager(Config.db_url())
    return _db
