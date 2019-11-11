import os
import logging
from dotenv import load_dotenv
from util.logger import init_logger


load_dotenv()

def test_logger():
    job_id = 'job_123'
    os.environ['JOB_ID'] = job_id

    assert job_id not in logging.root.manager.loggerDict, "job id is not initialized"

    init_logger()
    assert job_id in logging.root.manager.loggerDict, "job id is initialized"
