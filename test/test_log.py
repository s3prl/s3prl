import logging

from s3prl.util import Log, LogDataType

logger = logging.getLogger(__name__)

def test_log():
    logs = [
        Log("single scalar", 3, LogDataType.SCALAR),
        Log("scalar lists", 4, LogDataType.SCALAR),
        Log("text", "hellow", LogDataType.TEXT),
    ]
    for log in logs:
        logger.info(log.name)
        logger.info(log.data)
        logger.info(log.data_type)
