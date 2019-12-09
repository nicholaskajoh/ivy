import logging
from unittest.mock import patch
from dotenv import load_dotenv
from util.logger import init_logger, get_logger


load_dotenv()
job_id = 'job_123'

@patch.dict('os.environ', {'JOB_ID': job_id})
def test_init_logger():
    assert job_id not in logging.root.manager.loggerDict, 'job id is not initialized'

    init_logger()
    assert job_id in logging.root.manager.loggerDict, 'job id is initialized'

@patch.dict('os.environ', {'JOB_ID': job_id})
def test_get_logger():
    logger = get_logger()
    assert logger == logging.getLogger('job_123'), 'logger instance is retrieved'
