import logging

from s3prl import Logs

logger = logging.getLogger(__name__)


def test_log():
    logs = Logs()
    logs.add_scalar("loss", 4)
    for log in logs.values():
        logger.info(log.name)
        logger.info(log.data)
        logger.info(log.data_type)
