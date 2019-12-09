import os
from unittest.mock import patch
from util.job import get_job_id


def test_create_job_id():
    assert get_job_id() == os.getenv('JOB_ID'), 'job id is created'

@patch.dict('os.environ', {'JOB_ID': 'job_123'})
def test_retrieve_job_id():
    assert get_job_id() == 'job_123', 'job id is retrieved'
