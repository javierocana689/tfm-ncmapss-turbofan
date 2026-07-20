# config.py — configuración compartida de todos los notebooks del TFM
from pathlib import Path

DATA_DIR = Path(r"C:\Users\11jav\Documents\Master\TFM\Dataset_Nasa\Turbofan_Engine_degradation\Datasets\data_set\data_set")

FILES = {
    'DS01': DATA_DIR / 'N-CMAPSS_DS01-005.h5',
    'DS02': DATA_DIR / 'N-CMAPSS_DS02-006.h5',
    'DS03': DATA_DIR / 'N-CMAPSS_DS03-012.h5',
    'DS04': DATA_DIR / 'N-CMAPSS_DS04.h5',
    'DS05': DATA_DIR / 'N-CMAPSS_DS05.h5',
    'DS06': DATA_DIR / 'N-CMAPSS_DS06.h5',
    'DS07': DATA_DIR / 'N-CMAPSS_DS07.h5',
    'DS08a': DATA_DIR / 'N-CMAPSS_DS08a-009.h5',
    'DS08c': DATA_DIR / 'N-CMAPSS_DS08c-008.h5',
}