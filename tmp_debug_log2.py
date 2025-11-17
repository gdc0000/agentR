from survey_to_r.utils import setup_logging
import tempfile, os

tmp = tempfile.mkdtemp()
test_log=os.path.join(tmp,'test_log.jsonl')
print('creating', test_log)
setup_logging(test_log)
print('exists after setup', os.path.exists(test_log))
