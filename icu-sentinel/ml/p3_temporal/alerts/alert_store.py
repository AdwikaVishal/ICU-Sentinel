"""SQLite persistence for alert episodes.

The store is deliberately small and dependency-free. It keeps alert
history across application restarts and separates persistence from the
alert decision logic in :mod:`alert_engine`.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .alert_schemas import AlertRecord, AlertStatus


class AlertNotFoundError(LookupError):
    """Raised when an alert ID does not exist."""


class AlertStateError(ValueError):
    """Raised when an alert lifecycle transition is not permitted."""


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    hour INTEGER NOT NULL,
    first_hour INTEGER NOT NULL,
    last_hour INTEGER NOT NULL,
    probability REAL NOT NULL,
    peak_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    threshold_source TEXT NOT NULL,
    threshold_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_source TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    acknowledged_at TEXT,
    cleared_at TEXT,
    cleared_reason TEXT,
    occurrence_count INTEGER NOT NULL,
    is_synthetic INTEGER NOT NULL
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_episode ON alerts(patient_id, rule_id, status)",
)

_UPDATABLE_FIELDS = {
    "hour",
    "last_hour",
    "probability",
    "peak_probability",
    "threshold",
    "threshold_source",
    "threshold_version",
    "model_name",
    "model_version",
    "prediction_source",
    "status",
    "updated_at",
    "acknowledged_at",
    "cleared_at",
    "cleared_reason",
    "occurrence_count",
}


class AlertStore:
    """Thread-safe SQLite alert store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute(_CREATE_TABLE)
            for statement in _INDEXES:
                self._conn.execute(statement)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Create / read
    # ------------------------------------------------------------------
    def create_alert(self, record: AlertRecord) -> AlertRecord:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, patient_id, rule_id, hour, first_hour, last_hour,
                    probability, peak_probability, threshold, threshold_source,
                    threshold_version, model_name, model_version,
                    prediction_source, status, created_at, updated_at,
                    acknowledged_at, cleared_at, cleared_reason,
                    occurrence_count, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.alert_id,
                    record.patient_id,
                    record.rule_id,
                    record.hour,
                    record.first_hour,
                    record.last_hour,
                    record.probability,
                    record.peak_probability,
                    record.threshold,
                    record.threshold_source,
                    record.threshold_version,
                    record.model_name,
                    record.model_version,
                    record.prediction_source,
                    record.status.value,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.acknowledged_at.isoformat() if record.acknowledged_at else None,
                    record.cleared_at.isoformat() if record.cleared_at else None,
                    record.cleared_reason,
                    record.occurrence_count,
                    int(record.is_synthetic),
                ),
            )
            self._conn.commit()
        return record

    def get_alert(self, alert_id: str) -> AlertRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
        if row is None:
            raise AlertNotFoundError(f"Alert not found: {alert_id}")
        return _row_to_record(row)

    def list_alerts(
        self,
        *,
        status: AlertStatus | str | None = None,
        patient_id: str | None = None,
        limit: int = 200,
    ) -> list[AlertRecord]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(AlertStatus(status).value if not isinstance(status, AlertStatus) else status.value)
        if patient_id is not None:
            clauses.append("patient_id = ?")
            params.append(patient_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT * FROM alerts "
            f"{where} ORDER BY last_hour DESC, updated_at DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_active(self, *, limit: int = 200) -> list[AlertRecord]:
        return self.list_alerts(status=AlertStatus.ACTIVE, limit=limit)

    def list_history(self, *, limit: int = 200) -> list[AlertRecord]:
        """Return acknowledged and cleared episodes, newest first."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM alerts
                WHERE status IN (?, ?)
                ORDER BY last_hour DESC, updated_at DESC
                LIMIT ?
                """,
                (AlertStatus.ACKNOWLEDGED.value, AlertStatus.CLEARED.value, int(limit)),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_active_for_episode(
        self,
        patient_id: str,
        rule_id: str,
    ) -> AlertRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM alerts
                WHERE patient_id = ? AND rule_id = ? AND status IN (?, ?)
                ORDER BY last_hour DESC, updated_at DESC
                LIMIT 1
                """,
                (
                    patient_id,
                    rule_id,
                    AlertStatus.ACTIVE.value,
                    AlertStatus.ACKNOWLEDGED.value,
                ),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    # ------------------------------------------------------------------
    # Lifecycle updates
    # ------------------------------------------------------------------
    def update_alert(self, alert_id: str, **fields: Any) -> AlertRecord:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update alert fields: {sorted(unknown)}")
        if not fields:
            return self.get_alert(alert_id)

        prepared: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "status":
                prepared[key] = AlertStatus(value).value if not isinstance(value, AlertStatus) else value.value
            elif isinstance(value, datetime):
                prepared[key] = value.isoformat()
            elif key == "is_synthetic":
                prepared[key] = int(bool(value))
            else:
                prepared[key] = value

        assignments = ", ".join(f"{key} = ?" for key in prepared)
        params = list(prepared.values()) + [alert_id]
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE alerts SET {assignments} WHERE alert_id = ?",
                params,
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise AlertNotFoundError(f"Alert not found: {alert_id}")
        return self.get_alert(alert_id)

    def acknowledge_alert(self, alert_id: str, *, at: datetime) -> AlertRecord:
        try:
            record = self.get_alert(alert_id)
        except AlertNotFoundError:
            raise
        if record.status is AlertStatus.CLEARED:
            raise AlertStateError("A cleared alert cannot be acknowledged")
        if record.status is AlertStatus.ACKNOWLEDGED:
            return record
        return self.update_alert(
            alert_id,
            status=AlertStatus.ACKNOWLEDGED,
            acknowledged_at=at,
            updated_at=at,
        )

    def clear_alert(
        self,
        alert_id: str,
        *,
        at: datetime,
        reason: str,
    ) -> AlertRecord:
        try:
            record = self.get_alert(alert_id)
        except AlertNotFoundError:
            raise
        if record.status is AlertStatus.CLEARED:
            return record
        return self.update_alert(
            alert_id,
            status=AlertStatus.CLEARED,
            cleared_at=at,
            cleared_reason=reason,
            updated_at=at,
        )


def _row_to_record(row: sqlite3.Row) -> AlertRecord:
    data = dict(row)
    data["status"] = AlertStatus(data["status"])
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
    if data.get("acknowledged_at"):
        data["acknowledged_at"] = datetime.fromisoformat(data["acknowledged_at"])
    if data.get("cleared_at"):
        data["cleared_at"] = datetime.fromisoformat(data["cleared_at"])
    data["is_synthetic"] = bool(data["is_synthetic"])
    return AlertRecord(**data)
