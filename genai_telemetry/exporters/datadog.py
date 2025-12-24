"""
Datadog exporter for direct API integration.
"""

import atexit
import json
import logging
import threading
import time
import uuid
import urllib.request
from typing import Any, Dict, List, Optional

from genai_telemetry.exporters.base import BaseExporter

logger = logging.getLogger("genai_telemetry.exporters.datadog")


class DatadogExporter(BaseExporter):
    """Sends spans directly to Datadog API."""
    
    def __init__(
        self,
        api_key: str,
        site: str = "datadoghq.com",
        service_name: str = "genai-app",
        env: str = "production",
        batch_size: int = 10,
        flush_interval: float = 5.0
    ):
        self.api_key = api_key
        self.site = site
        self.service_name = service_name
        self.env = env
        self.endpoint = f"https://http-intake.logs.{site}/api/v2/logs"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._batch: List[dict] = []
        self._lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        
        atexit.register(self.stop)
    
    def start(self) -> None:
        """Start the exporter and background flush thread."""
        if self._running:
            return
        self._running = True
        if self.batch_size > 1:
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()
    
    def stop(self) -> None:
        """Stop the exporter and flush remaining data."""
        self._running = False
        self.flush()
    
    def _flush_loop(self) -> None:
        """Background thread for periodic flushing."""
        while self._running:
            time.sleep(self.flush_interval)
            self.flush()
    
    def flush(self) -> None:
        """Flush any buffered spans."""
        with self._lock:
            if not self._batch:
                return
            batch = self._batch.copy()
            self._batch = []
        self._send_batch(batch)
    
    def _send_batch(self, batch: List[dict]) -> bool:
        """Send a batch of spans to Datadog Logs API."""
        if not batch:
            return True
        
        dd_logs = []
        for span in batch:
            dd_logs.append({
                "ddsource": "genai-telemetry",
                "ddtags": f"env:{self.env},service:{self.service_name},span_type:{span.get('span_type', 'unknown')},model:{span.get('model_name', 'unknown')}",
                "hostname": span.get("workflow_name", "genai-app"),
                "service": self.service_name,
                "message": json.dumps(span)
            })
        
        payload = json.dumps(dd_logs)
        data = payload.encode("utf-8")
        
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": self.api_key,
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in [200, 202]
        except Exception as e:
            logger.error(f"Datadog Error: {e}")
            return False
    
    def export(self, span_data: Dict[str, Any]) -> bool:
        """Export a single span."""
        if self.batch_size <= 1:
            return self._send_batch([span_data])
        
        with self._lock:
            self._batch.append(span_data)
            should_flush = len(self._batch) >= self.batch_size
        
        if should_flush:
            self.flush()
        return True
