from docai.ingestion.track_b.graph_writer import FalkorWriter
import logging

logger = logging.getLogger(__name__)

async def graph_write(subject: str, predicate: str, object: str) -> str:
    """
    Write a fully validated relationship to the graph.
    """
    writer = FalkorWriter()
    try:
        writer.write_relation({
            "subject": subject,
            "predicate": predicate,
            "object": object
        })
        return "{\"status\": \"success\"}"
    except Exception as e:
        logger.error(f"Graph write failed: {e}")
        return f"{{\"status\": \"error\", \"detail\": \"{str(e)}\"}}"

async def quarantine_write(subject: str, predicate: str, object: str, reason: str) -> str:
    """
    Send a failed extraction to the quarantine queue.
    """
    writer = FalkorWriter()
    writer.write_quarantine({
        "subject": subject,
        "predicate": predicate,
        "object": object
    }, reason)
    return "{\"status\": \"quarantined\"}"

async def audit_log(action: str, entity_id: str, detail: str) -> str:
    """
    Write an audit log event.
    """
    logger.info(f"AUDIT LOG: {action} on {entity_id} - {detail}")
    return "{\"status\": \"logged\"}"

