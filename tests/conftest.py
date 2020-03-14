'''
Configure tests.
'''

from dotenv import load_dotenv


def pytest_configure():
    '''
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    '''
    load_dotenv()
