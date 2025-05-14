from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from core.signature_comparator import SignatureComparator
from utils.results_logger import ResultsLogger
from utils.user_registry import UserRegistry

import cv2
import os
import sqlite3


ur = UserRegistry()
ur.list_users()



