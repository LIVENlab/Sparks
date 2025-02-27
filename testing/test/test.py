import sys
import os
import pytest
from Sparks.util.base import SoftLink


pass


def test_init():
    test_path  = r'C:\Users\altz7\PycharmProjects\ENBIOS4TIMES_\testing\data_test'
    project_name =  'Seeds_exp4'

    softlink_instance = SoftLink(file_path=test_path, project=project_name)

    assert softlink_instance.project == project_name
    assert softlink_instance.file_path == test_path
    assert softlink_instance.SoftLink is None







if __name__ == '__main__':
    pytest.main([__file__])