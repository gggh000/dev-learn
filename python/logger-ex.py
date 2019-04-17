import logging

logger=logging.getLogger('test-app')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('test-app-debug.log')
ch = logging.StreamHandler()

fh.setLevel(logging.DEBUG)
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info("test-app-log emitted.")
logger.error("ERROR: test-app-log emitted.")
logger.critical("CRITICAL: test-app-log emitted.")

