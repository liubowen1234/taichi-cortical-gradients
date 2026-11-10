"""Colour palette of the main figures (colours follow Sydnor et al., 2025, Nat. Neurosci. 28, 1772).
Every colour used in the figures comes from this module."""
from matplotlib.colors import LinearSegmentedColormap as LSC

# ---- groups (Fig. 4f early/late: yellow #FEC22F vs navy #323280; grey #989898 = neutral / CI in all figures)
GROUPCOL = {'HC': '#989898', 'TC': '#FEC22F', 'BB': '#323280'}
GROUPEDGE = {'HC': '#6B6B6B', 'TC': '#C28F00', 'BB': '#1F1F5C'}     # darker outlines (yellow is faint on white)

# ---- three contrasts (colour + marker); colours from the Fig. 4a / 8a trajectory ramp
CONTRAST = {'TCvsHC': ('#FCAB6A', 'o'), 'BBvsHC': ('#323280', 's'), 'TCvsBB': ('#9B3A80', '^')}

# ---- gradient positions (G1/G2/G3, reference maps): sequential ramp of Fig. 1a / 4a (yellow -> indigo)
SEQ_HEX = ['#FEC22F', '#FBB050', '#F69E6C', '#E98D75', '#D2737A', '#A7457F', '#7C2F79', '#4F2F7B', '#323280']
SEQ = LSC.from_list('sydnor_seq', SEQ_HEX)

# ---- signed effects (g maps, partial r maps, network heat maps, lollipops): S-A axis map of Fig. 3c / 5b / 7d,
#      negative = yellow, 0 = near white, positive = purple
DIV_HEX = ['#FFC125', '#FFD16B', '#FFE7B4', '#F8F3F9', '#D0B0D4', '#A064AA', '#7E308E', '#571766']
DIV = LSC.from_list('sydnor_div', DIV_HEX)

# ---- non-negative magnitudes (edge counts, |t|): lavender -> navy of Fig. 6 maps
MAG_HEX = ['#F1F1F7', '#A2A2C8', '#323280']
MAG = LSC.from_list('sydnor_mag', MAG_HEX)

# ---- significance and neutrals (Fig. 5e / 7e squares; grey regions; CI band)
SIG, NS = '#672975', '#A2A2C8'
EQUIV, GRID, CIBAND, NA = '#EDEDED', '#E5E5E5', '#989898', '#E3E3E3'
INK = '#000000'

# ---- NBS chord edges
EDGE = {'VIS2-SMN': '#E98D75', 'VIS2-CON': '#672975', 'VIS2-other': '#A2A2C8', 'other': '#E5E5E5'}
