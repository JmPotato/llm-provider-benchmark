from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from providers import ProviderConfig
from records import RequestRecord, TokenEvent

logger = logging.getLogger(__name__)


class BenchmarkStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        logger.debug("Opening database: %s", db_path)
        self.connection = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        logger.debug("Initializing schema")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                started_at DOUBLE NOT NULL,
                finished_at DOUBLE,
                duration_s DOUBLE,
                config_json VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS providers (
                run_id VARCHAR NOT NULL,
                provider_name VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                api_base VARCHAR,
                api_key_env VARCHAR,
                extra_headers_json VARCHAR NOT NULL,
                temperature DOUBLE,
                max_tokens BIGINT,
                timeout_s DOUBLE,
                PRIMARY KEY (run_id, provider_name)
            );

            CREATE TABLE IF NOT EXISTS requests (
                run_id VARCHAR NOT NULL,
                provider_name VARCHAR NOT NULL,
                request_id VARCHAR NOT NULL,
                prompt VARCHAR NOT NULL,
                request_started_at DOUBLE NOT NULL,
                first_token_at DOUBLE,
                response_done_at DOUBLE NOT NULL,
                output_tokens BIGINT NOT NULL,
                success BOOLEAN NOT NULL,
                error_type VARCHAR,
                error_message VARCHAR,
                ttft_s DOUBLE,
                tbt_s DOUBLE,
                tpot_s DOUBLE,
                e2e_s DOUBLE NOT NULL,
                slo_passed BOOLEAN NOT NULL,
                PRIMARY KEY (run_id, provider_name, request_id)
            );

            CREATE TABLE IF NOT EXISTS token_events (
                run_id VARCHAR NOT NULL,
                provider_name VARCHAR NOT NULL,
                request_id VARCHAR NOT NULL,
                token_index BIGINT NOT NULL,
                token_timestamp DOUBLE NOT NULL,
                token_text VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS window_metrics_1s (
                run_id VARCHAR NOT NULL,
                provider_name VARCHAR NOT NULL,
                window_start BIGINT NOT NULL,
                request_count BIGINT NOT NULL,
                output_tokens BIGINT NOT NULL,
                rps DOUBLE NOT NULL,
                tps DOUBLE NOT NULL,
                goodput DOUBLE NOT NULL,
                error_rate DOUBLE NOT NULL,
                PRIMARY KEY (run_id, provider_name, window_start)
            );
            """
        )

    def create_run(self, run_id: str, started_at: float, config_json: str) -> None:
        logger.debug("Creating run: %s", run_id)
        self.connection.execute(
            """
            INSERT INTO runs (run_id, started_at, config_json)
            VALUES (?, ?, ?)
            """,
            [run_id, started_at, config_json],
        )

    def finish_run(self, run_id: str, finished_at: float) -> None:
        logger.debug("Finishing run: %s", run_id)
        self.connection.execute(
            """
            UPDATE runs
            SET finished_at = ?, duration_s = ? - started_at
            WHERE run_id = ?
            """,
            [finished_at, finished_at, run_id],
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT run_id, started_at, finished_at, duration_s, config_json
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {
            "run_id": row[0],
            "started_at": row[1],
            "finished_at": row[2],
            "duration_s": row[3],
            "config_json": row[4],
        }

    def delete_run(self, run_id: str) -> bool:
        existing = self.connection.execute(
            """
            SELECT 1
            FROM runs
            WHERE run_id = ?
            LIMIT 1
            """,
            [run_id],
        ).fetchone()
        if existing is None:
            return False

        logger.debug("Deleting run: %s", run_id)
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                "DELETE FROM token_events WHERE run_id = ?", [run_id]
            )
            self.connection.execute(
                "DELETE FROM window_metrics_1s WHERE run_id = ?", [run_id]
            )
            self.connection.execute("DELETE FROM requests WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM providers WHERE run_id = ?", [run_id])
            self.connection.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
            self.connection.execute("COMMIT")
        except Exception:  # noqa: BLE001
            logger.warning("Error during deletion of run %s, rolling back transaction", run_id, exc_info=True)
            self.connection.execute("ROLLBACK")
            raise
        return True

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT run_id, started_at, finished_at, duration_s, config_json
            FROM runs
            ORDER BY started_at DESC
            """
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "started_at": row[1],
                "finished_at": row[2],
                "duration_s": row[3],
                "config_json": row[4],
            }
            for row in rows
        ]

    def list_runs_with_stats(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            WITH provider_counts AS (
                SELECT run_id, COUNT(*) AS provider_count
                FROM providers
                GROUP BY run_id
            ),
            request_counts AS (
                SELECT
                    run_id,
                    COUNT(*) AS request_count,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN success THEN 0 ELSE 1 END) AS failed_count
                FROM requests
                GROUP BY run_id
            )
            SELECT
                runs.run_id,
                runs.started_at,
                runs.finished_at,
                runs.duration_s,
                runs.config_json,
                COALESCE(provider_counts.provider_count, 0) AS provider_count,
                COALESCE(request_counts.request_count, 0) AS request_count,
                COALESCE(request_counts.success_count, 0) AS success_count,
                COALESCE(request_counts.failed_count, 0) AS failed_count
            FROM runs
            LEFT JOIN provider_counts USING (run_id)
            LEFT JOIN request_counts USING (run_id)
            ORDER BY runs.started_at DESC
            """
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "started_at": row[1],
                "finished_at": row[2],
                "duration_s": row[3],
                "config_json": row[4],
                "provider_count": int(row[5] or 0),
                "request_count": int(row[6] or 0),
                "success_count": int(row[7] or 0),
                "failed_count": int(row[8] or 0),
            }
            for row in rows
        ]

    def upsert_provider_snapshot(self, run_id: str, provider: ProviderConfig) -> None:
        self.connection.execute(
            """
            INSERT INTO providers (
                run_id,
                provider_name,
                model,
                api_base,
                api_key_env,
                extra_headers_json,
                temperature,
                max_tokens,
                timeout_s
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, provider_name) DO UPDATE SET
                model = excluded.model,
                api_base = excluded.api_base,
                api_key_env = excluded.api_key_env,
                extra_headers_json = excluded.extra_headers_json,
                temperature = excluded.temperature,
                max_tokens = excluded.max_tokens,
                timeout_s = excluded.timeout_s
            """,
            [
                run_id,
                provider.name,
                provider.model,
                provider.api_base,
                provider.api_key_env,
                json.dumps(provider.extra_headers, ensure_ascii=True),
                provider.temperature,
                provider.max_tokens,
                provider.timeout_s,
            ],
        )

    def list_run_providers(self, run_id: str) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT provider_name
            FROM providers
            WHERE run_id = ?
            ORDER BY provider_name ASC
            """,
            [run_id],
        ).fetchall()
        return [row[0] for row in rows]

    def insert_request_records(self, records: list[RequestRecord]) -> None:
        if not records:
            return
        logger.debug("Inserting %d request records", len(records))
        self.connection.executemany(
            """
            INSERT INTO requests (
                run_id,
                provider_name,
                request_id,
                prompt,
                request_started_at,
                first_token_at,
                response_done_at,
                output_tokens,
                success,
                error_type,
                error_message,
                ttft_s,
                tbt_s,
                tpot_s,
                e2e_s,
                slo_passed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.run_id,
                    record.provider_name,
                    record.request_id,
                    record.prompt,
                    record.request_started_at,
                    record.first_token_at,
                    record.response_done_at,
                    record.output_tokens,
                    record.success,
                    record.error_type,
                    record.error_message,
                    record.ttft_s,
                    record.tbt_s,
                    record.tpot_s,
                    record.e2e_s,
                    record.slo_passed,
                )
                for record in records
            ],
        )

    def insert_token_events(self, events: list[TokenEvent]) -> None:
        if not events:
            return
        logger.debug("Inserting %d token events", len(events))
        self.connection.executemany(
            """
            INSERT INTO token_events (
                run_id,
                provider_name,
                request_id,
                token_index,
                token_timestamp,
                token_text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event.run_id,
                    event.provider_name,
                    event.request_id,
                    event.token_index,
                    event.token_timestamp,
                    event.token_text,
                )
                for event in events
            ],
        )

    def refresh_window_metrics(self, run_id: str, provider_name: str) -> None:
        logger.debug("Refreshing window metrics: run=%s provider=%r", run_id, provider_name)
        self.connection.execute(
            """
            DELETE FROM window_metrics_1s
            WHERE run_id = ? AND provider_name = ?
            """,
            [run_id, provider_name],
        )
        self.connection.execute(
            """
            INSERT INTO window_metrics_1s (
                run_id,
                provider_name,
                window_start,
                request_count,
                output_tokens,
                rps,
                tps,
                goodput,
                error_rate
            )
            SELECT
                run_id,
                provider_name,
                CAST(FLOOR(response_done_at) AS BIGINT) AS window_start,
                COUNT(*) AS request_count,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COUNT(*)::DOUBLE AS rps,
                COALESCE(SUM(output_tokens), 0)::DOUBLE AS tps,
                SUM(CASE WHEN slo_passed THEN 1 ELSE 0 END)::DOUBLE AS goodput,
                SUM(CASE WHEN success THEN 0 ELSE 1 END)::DOUBLE / COUNT(*)::DOUBLE AS error_rate
            FROM requests
            WHERE run_id = ? AND provider_name = ?
            GROUP BY run_id, provider_name, window_start
            ORDER BY window_start
            """,
            [run_id, provider_name],
        )

    def get_provider_summary(self, run_id: str, provider_name: str) -> dict[str, Any]:
        provider_exists = self.connection.execute(
            """
            SELECT 1
            FROM providers
            WHERE run_id = ? AND provider_name = ?
            LIMIT 1
            """,
            [run_id, provider_name],
        ).fetchone()
        if provider_exists is None:
            raise KeyError(f"{run_id}/{provider_name}")

        latency = {
            "ttft": self._quantiles_from_requests(run_id, provider_name, "ttft_s"),
            "tpot": self._quantiles_from_requests(run_id, provider_name, "tpot_s"),
            "tbt": self._quantiles_from_requests(run_id, provider_name, "tbt_s"),
            "e2e": self._quantiles_from_requests(run_id, provider_name, "e2e_s"),
            "itl": self._quantiles_from_itl(run_id, provider_name),
        }
        throughput = {
            "rps": self._quantiles_from_windows(run_id, provider_name, "rps"),
            "tps": self._quantiles_from_windows(run_id, provider_name, "tps"),
        }
        quality = {
            "goodput": self._quantiles_from_windows(run_id, provider_name, "goodput"),
            "error_rate": self._quantiles_from_windows(
                run_id, provider_name, "error_rate"
            ),
        }
        return {"latency": latency, "throughput": throughput, "quality": quality}

    def close(self) -> None:
        self.connection.close()

    def _quantiles_from_requests(
        self, run_id: str, provider_name: str, column_name: str
    ) -> dict[str, float | int | None]:
        row = self.connection.execute(
            f"""
            SELECT
                COUNT({column_name}) AS count_value,
                quantile_cont({column_name}, 0.5) AS p50,
                quantile_cont({column_name}, 0.9) AS p90,
                quantile_cont({column_name}, 0.95) AS p95,
                quantile_cont({column_name}, 0.99) AS p99
            FROM requests
            WHERE run_id = ? AND provider_name = ? AND {column_name} IS NOT NULL
            """,
            [run_id, provider_name],
        ).fetchone()
        return self._row_to_quantile_dict(row)

    def _quantiles_from_windows(
        self, run_id: str, provider_name: str, column_name: str
    ) -> dict[str, float | int | None]:
        row = self.connection.execute(
            f"""
            SELECT
                COUNT({column_name}) AS count_value,
                quantile_cont({column_name}, 0.5) AS p50,
                quantile_cont({column_name}, 0.9) AS p90,
                quantile_cont({column_name}, 0.95) AS p95,
                quantile_cont({column_name}, 0.99) AS p99
            FROM window_metrics_1s
            WHERE run_id = ? AND provider_name = ? AND {column_name} IS NOT NULL
            """,
            [run_id, provider_name],
        ).fetchone()
        return self._row_to_quantile_dict(row)

    def _quantiles_from_itl(
        self, run_id: str, provider_name: str
    ) -> dict[str, float | int | None]:
        row = self.connection.execute(
            """
            WITH token_gaps AS (
                SELECT
                    token_timestamp
                    - LAG(token_timestamp) OVER (
                        PARTITION BY run_id, provider_name, request_id
                        ORDER BY token_index
                    ) AS itl_s
                FROM token_events
                WHERE run_id = ? AND provider_name = ?
            )
            SELECT
                COUNT(itl_s) AS count_value,
                quantile_cont(itl_s, 0.5) AS p50,
                quantile_cont(itl_s, 0.9) AS p90,
                quantile_cont(itl_s, 0.95) AS p95,
                quantile_cont(itl_s, 0.99) AS p99
            FROM token_gaps
            WHERE itl_s IS NOT NULL
            """,
            [run_id, provider_name],
        ).fetchone()
        return self._row_to_quantile_dict(row)

    @staticmethod
    def _row_to_quantile_dict(row: Any) -> dict[str, float | int | None]:
        if row is None:
            return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None}
        return {
            "count": int(row[0] or 0),
            "p50": (float(row[1]) if row[1] is not None else None),
            "p90": (float(row[2]) if row[2] is not None else None),
            "p95": (float(row[3]) if row[3] is not None else None),
            "p99": (float(row[4]) if row[4] is not None else None),
        }
