import os
from util.job import get_job_id


def test_get_job_id():
    job_id = get_job_id()
    assert job_id == os.getenv('JOB_ID'), "job id is created"

    os.environ['JOB_ID'] = 'job_123'
    assert get_job_id() == 'job_123', "job id is retrieved"
