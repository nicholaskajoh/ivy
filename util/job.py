'''
Utilities for managing object count jobs.
'''

import os
import uuid
import time


def get_job_id():
    '''
    Fetch job id or create new if one doesn't already exist.
    '''
    if os.getenv('JOB_ID') is None:
        os.environ['JOB_ID'] = 'job_' + str(int(time.time())) + '_' + uuid.uuid4().hex
    return os.getenv('JOB_ID')
