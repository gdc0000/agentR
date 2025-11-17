from survey_to_r.utils import log_session
import tempfile, os
from unittest.mock import patch, MagicMock

tmp = tempfile.mkdtemp()
test_log=os.path.join(tmp,'test_log.jsonl')
mock=MagicMock()

def cfg_get(k, d=None):
    if k=='log_file':
        return test_log
    if k=='mask_file_names':
        return True
    return d

mock.get.side_effect = cfg_get

with patch('survey_to_r.utils.config', mock):
    log_session({'event':'e'})
    print('exists', os.path.exists(test_log))
    print('content length', len(open(test_log).read()) if os.path.exists(test_log) else 0)
