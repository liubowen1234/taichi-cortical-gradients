"""Read config/config.json and define where every pipeline step writes its results.

All scripts import this module instead of writing paths by hand:

    from common.config import CFG, LAYOUT

CFG     -- the settings in config/config.json (input folders, external tools, analysis options).
LAYOUT  -- the output folders of all steps (one place, so every step finds the files of the step before it).

Relative paths in config.json are interpreted relative to the folder that holds config.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the code_release folder
CONFIG_FILE = ROOT / 'config' / 'config.json'
RESOURCES = ROOT / 'resources'


class _Config:
    def __init__(self, file):
        if not file.exists():
            raise FileNotFoundError(f'{file} not found. Copy config/config_template.json to config/config.json and edit the paths.')
        self.raw = json.loads(file.read_text(encoding='utf-8'))
        self.base = file.parent

    def path(self, key):
        """Absolute path of an entry under "paths" (None if the entry is empty)."""
        v = self.raw['paths'].get(key)
        if v in (None, ''):
            return None
        p = Path(v).expanduser()
        return p if p.is_absolute() else (self.base / p).resolve()

    def datasets(self, pipeline):
        """List of XCP-D output folders for a denoising pipeline ('36P' or '32P'), each with its FC source.
        fc_source = 'pconn'    : use the parcellated connectivity file written by XCP-D;
        fc_source = 'dtseries' : compute the connectivity from the denoised (unsmoothed) CIFTI time series."""
        out = []
        for d in self.raw['paths'][f'xcpd_{pipeline}']:
            p = Path(d['dir']).expanduser()
            out.append(dict(dir=p if p.is_absolute() else (self.base / p).resolve(), fc_source=d.get('fc_source', 'dtseries')))
        return out

    def option(self, key):
        return self.raw['analysis'][key]


CFG = _Config(CONFIG_FILE)
OUT = CFG.path('output_root')


class LAYOUT:
    """Output folders. Step numbers follow the README."""
    COHORT = OUT / '01_cohort'                                     # participant table, motion, FC arrays
    FC = COHORT / 'fc'
    GRAD = OUT / '02_gradients'
    GRAD_A = GRAD / 'glasser358_HCreference'                       # main analysis (MATLAB steps 1-4 write here)
    GRAD_B = GRAD / 'glasser358_external_template'                 # alignment to the BrainSpace template
    NET = OUT / '03_network_connectivity'
    SEG = NET / 'segregation'
    VIS2 = NET / 'visual2_connectivity'
    NBS = NET / 'nbs'
    SPATIAL = OUT / '04_spatial'
    SPIN = SPATIAL / 'spin_permutations'
    CONTEXT = SPATIAL / 'map_contextualization'
    SUBCORT = SPATIAL / 'subcortical_projection'
    DOSE = OUT / '05_training_dose'
    ROB = OUT / '06_robustness'
    MOTION = ROB / 'motion_subsets'
    QUALITY = ROB / 'data_quality'
    GSA = ROB / 'global_signal_amplitude'
    NOGSR = ROB / 'no_GSR_32P'
    GRAD_NOGSR = NOGSR / 'glasser358_HCreference'
    SCHAEFER = {400: ROB / 'schaefer400', 600: ROB / 'schaefer600'}
    MULTIVERSE = ROB / 'multiverse'
    ALIGN = ROB / 'alignment_comparison'
    SUMMARY = ROB / 'robustness_summary'
    FIGURES = OUT / '07_figures'


def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)
