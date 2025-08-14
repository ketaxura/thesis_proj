# logger_setup.py
import logging
from datetime import datetime
import os

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# File name with date/time stamp
log_filename = datetime.now().strftime("logs/mpc_run_%Y%m%d_%H%M%S.log")

# Configure logging
logging.basicConfig(
    filename=log_filename,
    filemode="w",   # overwrite each run
    level=logging.DEBUG,  # capture DEBUG and above
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Console output handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
)

# Add console handler to root logger
logging.getLogger().addHandler(console_handler)

logging.info("=== New MPC Session Started ===")

# Export the logger object
logger = logging.getLogger()
