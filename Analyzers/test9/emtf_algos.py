"""Algorithms in EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_utils import *


# ______________________________________________________________________________
# Configs

# 2 endcaps, 6 sectors per endcap. 3 zones and 3 timezones per sector.
num_emtf_sectors = 12
num_emtf_zones = 3
num_emtf_timezones = 3

# chambers: 54 (CSC) + 43 (RPC) + 11 (GEM) + 4 (ME0)
# segments: 2 (CSC), 2 (RPC), 8 (GEM), 4 (iRPC), 20 (ME0) - using CSC as default
# variables: 13 (emtf_phi, emtf_bend, emtf_theta1, emtf_theta2, emtf_qual1, emtf_qual2, emtf_time, zones, timezones, fr, dl, bx, valid)
num_emtf_chambers = 115
num_emtf_segments = 2
num_emtf_variables = 13

num_emtf_sites = 12
num_emtf_sites_rm = 5
num_emtf_hosts = 19

num_emtf_tracks = 4
num_emtf_patterns = 7
num_emtf_features = 36 + 4

# Eta boundaries used to define the zones
emtf_eta_bins = (0.8, 1.2, 1.55, 1.98, 2.4)

# Full range of emtf_phi is assumed to be 0..5040 (0..84 deg).
# 84 deg from 60 (native) + 20 (neighbor) + 2 (tolerance, left) + 2 (tolerance, right).
coarse_emtf_strip = 8 * 2  # 'doublestrip' unit
num_coarse_emtf_strips = 288  # num of doublestrip units to allocate
min_emtf_strip = (315 - num_coarse_emtf_strips) * coarse_emtf_strip  # 7.2 deg
max_emtf_strip = (315 - 0) * coarse_emtf_strip  # 84 deg

# ______________________________________________________________________________
# Functions

# Encode EMTF site number
# Total: 12 (5 from CSC + 4 from RPC + 3 from GEM)
def find_emtf_site_initializer():
  default_value = -99
  lut = np.full((5,5,5), default_value, dtype=np.int32)  # (type, station, ring) -> site
  lut[1,1,4] = 0  # ME1/1a
  lut[1,1,1] = 0  # ME1/1b
  lut[1,1,2] = 1  # ME1/2
  lut[1,1,3] = 1  # ME1/3
  lut[1,2,1] = 2  # ME2/1
  lut[1,2,2] = 2  # ME2/2
  lut[1,3,1] = 3  # ME3/1
  lut[1,3,2] = 3  # ME3/2
  lut[1,4,1] = 4  # ME4/1
  lut[1,4,2] = 4  # ME4/2
  lut[2,1,2] = 5  # RE1/2
  lut[2,1,3] = 5  # RE1/3
  lut[2,2,2] = 6  # RE2/2
  lut[2,2,3] = 6  # RE2/3
  lut[2,3,1] = 7  # RE3/1
  lut[2,3,2] = 7  # RE3/2
  lut[2,3,3] = 7  # RE3/3
  lut[2,4,1] = 8  # RE4/1
  lut[2,4,2] = 8  # RE4/2
  lut[2,4,3] = 8  # RE4/3
  lut[3,1,1] = 9  # GE1/1
  lut[3,2,1] = 10 # GE2/1
  lut[4,1,1] = 11 # ME0

  def lookup(_type, station, ring):
    multi_index = np.array([_type, station, ring])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_site = find_emtf_site_initializer()

# Encode EMTF host number, which is more narrowly defined than site number
# Total: 19 (9 from CSC + 9 from GEM & RPC + 1 from ME0)
def find_emtf_host_initializer():
  default_value = -99
  lut = np.full((5,5,5), default_value, dtype=np.int32)  # (type, station, ring) -> host
  lut[1,1,4] = 0  # ME1/1a
  lut[1,1,1] = 0  # ME1/1b
  lut[1,1,2] = 1  # ME1/2
  lut[1,1,3] = 2  # ME1/3
  lut[1,2,1] = 3  # ME2/1
  lut[1,2,2] = 4  # ME2/2
  lut[1,3,1] = 5  # ME3/1
  lut[1,3,2] = 6  # ME3/2
  lut[1,4,1] = 7  # ME4/1
  lut[1,4,2] = 8  # ME4/2
  lut[3,1,1] = 9  # GE1/1
  lut[2,1,2] = 10 # RE1/2
  lut[2,1,3] = 11 # RE1/3
  lut[3,2,1] = 12 # GE2/1
  lut[2,2,2] = 13 # RE2/2
  lut[2,2,3] = 13 # RE2/3
  lut[2,3,1] = 14 # RE3/1
  lut[2,3,2] = 15 # RE3/2
  lut[2,3,3] = 15 # RE3/3
  lut[2,4,1] = 16 # RE4/1
  lut[2,4,2] = 17 # RE4/2
  lut[2,4,3] = 17 # RE4/3
  lut[4,1,1] = 18 # ME0

  def lookup(_type, station, ring):
    multi_index = np.array([_type, station, ring])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_host = find_emtf_host_initializer()

# Decode EMTF site number
def decode_emtf_site_initializer():
  lut_0 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4])  # exact
  lut_1 = np.array([1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1])  # exact
  lut_2 = np.array([1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1])  # inexact

  def lookup(emtf_site):
    _type = np.take(lut_0, emtf_site)
    station = np.take(lut_1, emtf_site)
    ring = np.take(lut_2, emtf_site)
    return (_type, station, ring)
  return lookup

# The initializer will instantiate the lookup tables
decode_emtf_site = decode_emtf_site_initializer()

# Decode EMTF host number
def decode_emtf_host_initializer():
  lut_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 4])  # exact
  lut_1 = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1])  # exact
  lut_2 = np.array([1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1])  # inexact

  def lookup(emtf_host):
    _type = np.take(lut_0, emtf_host)
    station = np.take(lut_1, emtf_host)
    ring = np.take(lut_2, emtf_host)
    return (_type, station, ring)
  return lookup

# The initializer will instantiate the lookup tables
decode_emtf_host = decode_emtf_host_initializer()

host_to_site_lut = np.array([
  0, 1, 1, 2, 2, 3, 3, 4, 4, 9, 5, 5, 10, 6, 7, 7, 8, 8, 11
], dtype=np.int32)

# Hack ME0 chamber number
# Converting from 20-deg chamber into 10-deg chamber
def hack_me0_hit_chamber(hit):
  if hit.type == kME0:
    old_chamber = hit.chamber
    new_chamber = (old_chamber - 1) * 2 + 2
    if hit.endcap == 1:  # positive endcap
      try:
        if hit.strip <= 191:  # first 5 deg (1/4 of chamber)
          pass
        elif 191 < hit.strip < (767 - 192):  # next 10 deg (1/2 of chamber)
          new_chamber -= 1
        elif hit.strip >= (767 - 192):  # last 5 deg (1/4 of chamber)
          new_chamber -= 2
      except:
        pass
    else:  # negative endcap
      try:
        if hit.strip >= (767 - 192):  # first 5 deg (1/4 of chamber)
          pass
        elif 191 < hit.strip < (767 - 192):  # next 10 deg (1/2 of chamber)
          new_chamber -= 1
        elif hit.strip <= 191:  # last 5 deg (1/4 of chamber)
          new_chamber -= 2
      except:
        pass
    if new_chamber == 0:
      new_chamber += 36

    # Overwrite chamber, sector, subsector, cscid
    hit.chamber = new_chamber
    hit.sector = get_trigger_sector(hit.ring, hit.station, hit.chamber)
    hit.subsector = get_trigger_subsector(hit.ring, hit.station, hit.chamber)
    hit.cscid = get_trigger_cscid(hit.ring, hit.station, hit.chamber)
    try:
      if hit.neighbor:
        get_next_sector = lambda sector: (sector + 1) if sector != 6 else (sector + 1 - 6)
        hit.sector = get_next_sector(hit.sector)
    except:
      pass
  return

# Encode EMTF chamber number
# Total: 112 (6*9*2 + 4)
def find_emtf_chamber_initializer():
  default_value = -99
  lut = np.full((5,5,10,4), default_value, dtype=np.int32)  # (type, station, cscid, subsector) -> chamber
  lut[1,1,1,1] = 0   # ME1/1 sub 1
  lut[1,1,2,1] = 1   # ME1/1 sub 1
  lut[1,1,3,1] = 2   # ME1/1 sub 1
  lut[1,1,4,1] = 3   # ME1/2 sub 1
  lut[1,1,5,1] = 4   # ME1/2 sub 1
  lut[1,1,6,1] = 5   # ME1/2 sub 1
  lut[1,1,7,1] = 6   # ME1/3 sub 1
  lut[1,1,8,1] = 7   # ME1/3 sub 1
  lut[1,1,9,1] = 8   # ME1/3 sub 1
  lut[1,1,1,2] = 9   # ME1/1 sub 2
  lut[1,1,2,2] = 10  # ME1/1 sub 2
  lut[1,1,3,2] = 11  # ME1/1 sub 2
  lut[1,1,4,2] = 12  # ME1/2 sub 2
  lut[1,1,5,2] = 13  # ME1/2 sub 2
  lut[1,1,6,2] = 14  # ME1/2 sub 2
  lut[1,1,7,2] = 15  # ME1/3 sub 2
  lut[1,1,8,2] = 16  # ME1/3 sub 2
  lut[1,1,9,2] = 17  # ME1/3 sub 2
  lut[1,2,1,0] = 18  # ME2/1
  lut[1,2,2,0] = 19  # ME2/1
  lut[1,2,3,0] = 20  # ME2/1
  lut[1,2,4,0] = 21  # ME2/2
  lut[1,2,5,0] = 22  # ME2/2
  lut[1,2,6,0] = 23  # ME2/2
  lut[1,2,7,0] = 24  # ME2/2
  lut[1,2,8,0] = 25  # ME2/2
  lut[1,2,9,0] = 26  # ME2/2
  lut[1,3,1,0] = 27  # ME3/1
  lut[1,3,2,0] = 28  # ME3/1
  lut[1,3,3,0] = 29  # ME3/1
  lut[1,3,4,0] = 30  # ME3/2
  lut[1,3,5,0] = 31  # ME3/2
  lut[1,3,6,0] = 32  # ME3/2
  lut[1,3,7,0] = 33  # ME3/2
  lut[1,3,8,0] = 34  # ME3/2
  lut[1,3,9,0] = 35  # ME3/2
  lut[1,4,1,0] = 36  # ME4/1
  lut[1,4,2,0] = 37  # ME4/1
  lut[1,4,3,0] = 38  # ME4/1
  lut[1,4,4,0] = 39  # ME4/2
  lut[1,4,5,0] = 40  # ME4/2
  lut[1,4,6,0] = 41  # ME4/2
  lut[1,4,7,0] = 42  # ME4/2
  lut[1,4,8,0] = 43  # ME4/2
  lut[1,4,9,0] = 44  # ME4/2
  lut[1,1,3,3] = 45  # ME1/1 neigh
  lut[1,1,6,3] = 46  # ME1/2 neigh
  lut[1,1,9,3] = 47  # ME1/3 neigh
  lut[1,2,3,3] = 48  # ME2/1 neigh
  lut[1,2,9,3] = 49  # ME2/2 neigh
  lut[1,3,3,3] = 50  # ME3/1 neigh
  lut[1,3,9,3] = 51  # ME3/2 neigh
  lut[1,4,3,3] = 52  # ME4/1 neigh
  lut[1,4,9,3] = 53  # ME4/2 neigh
  #
  lut[3,1,1,1] = 54  # GE1/1 sub 1
  lut[3,1,2,1] = 55  # GE1/1 sub 1
  lut[3,1,3,1] = 56  # GE1/1 sub 1
  lut[2,1,4,1] = 57  # RE1/2 sub 1
  lut[2,1,5,1] = 58  # RE1/2 sub 1
  lut[2,1,6,1] = 59  # RE1/2 sub 1
  lut[2,1,7,1] = 60  # RE1/3 sub 1
  lut[2,1,8,1] = 61  # RE1/3 sub 1
  lut[2,1,9,1] = 62  # RE1/3 sub 1
  lut[3,1,1,2] = 63  # GE1/1 sub 2
  lut[3,1,2,2] = 64  # GE1/1 sub 2
  lut[3,1,3,2] = 65  # GE1/1 sub 2
  lut[2,1,4,2] = 66  # RE1/2 sub 2
  lut[2,1,5,2] = 67  # RE1/2 sub 2
  lut[2,1,6,2] = 68  # RE1/2 sub 2
  lut[2,1,7,2] = 69  # RE1/3 sub 2
  lut[2,1,8,2] = 70  # RE1/3 sub 2
  lut[2,1,9,2] = 71  # RE1/3 sub 2
  lut[3,2,1,0] = 72  # GE2/1
  lut[3,2,2,0] = 73  # GE2/1
  lut[3,2,3,0] = 74  # GE2/1
  lut[2,2,4,0] = 75  # RE2/2
  lut[2,2,5,0] = 76  # RE2/2
  lut[2,2,6,0] = 77  # RE2/2
  lut[2,2,7,0] = 78  # RE2/2
  lut[2,2,8,0] = 79  # RE2/2
  lut[2,2,9,0] = 80  # RE2/2
  lut[2,3,1,0] = 81  # RE3/1
  lut[2,3,2,0] = 82  # RE3/1
  lut[2,3,3,0] = 83  # RE3/1
  lut[2,3,4,0] = 84  # RE3/2
  lut[2,3,5,0] = 85  # RE3/2
  lut[2,3,6,0] = 86  # RE3/2
  lut[2,3,7,0] = 87  # RE3/2
  lut[2,3,8,0] = 88  # RE3/2
  lut[2,3,9,0] = 89  # RE3/2
  lut[2,4,1,0] = 90  # RE4/1
  lut[2,4,2,0] = 91  # RE4/1
  lut[2,4,3,0] = 92  # RE4/1
  lut[2,4,4,0] = 93  # RE4/2
  lut[2,4,5,0] = 94  # RE4/2
  lut[2,4,6,0] = 95  # RE4/2
  lut[2,4,7,0] = 96  # RE4/2
  lut[2,4,8,0] = 97  # RE4/2
  lut[2,4,9,0] = 98  # RE4/2
  lut[3,1,3,3] = 99  # GE1/1 neigh
  lut[2,1,6,3] = 100 # RE1/2 neigh
  lut[2,1,9,3] = 101 # RE1/3 neigh
  lut[3,2,3,3] = 102 # GE2/1 neigh
  lut[2,2,9,3] = 103 # RE2/2 neigh
  lut[2,3,3,3] = 104 # RE3/1 neigh
  lut[2,3,9,3] = 105 # RE3/2 neigh
  lut[2,4,3,3] = 106 # RE4/1 neigh
  lut[2,4,9,3] = 107 # RE4/2 neigh
  #
  lut[4,1,1,1] = 108 # ME0 sub 1
  lut[4,1,2,1] = 109 # ME0 sub 1
  lut[4,1,3,1] = 110 # ME0 sub 1
  lut[4,1,1,2] = 111 # ME0 sub 2
  lut[4,1,2,2] = 112 # ME0 sub 2
  lut[4,1,3,2] = 113 # ME0 sub 2
  lut[4,1,2,3] = 114 # ME0 neigh
  lut[4,1,3,3] = 114 # ME0 neigh

  def lookup(_type, station, cscid, subsector, neighbor):
    subsector = np.where(neighbor, 3, subsector)  # neighbor -> subsector 3
    multi_index = np.array([_type, station, cscid, subsector])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_chamber = find_emtf_chamber_initializer()

# Decode EMTF chamber number
def decode_emtf_chamber():
  raise NotImplementedError()

chamber_to_host_lut = np.array([
   0, 0, 0, 1, 1, 1, 2, 2, 2,  # ME1/1 sub 1, ME1/2 sub 1, ME1/3 sub 1
   0, 0, 0, 1, 1, 1, 2, 2, 2,  # ME1/1 sub 2, ME1/2 sub 2, ME1/3 sub 2
   3, 3, 3, 4, 4, 4, 4, 4, 4,  # ME2/1, ME2/2
   5, 5, 5, 6, 6, 6, 6, 6, 6,  # ME3/1, ME3/2
   7, 7, 7, 8, 8, 8, 8, 8, 8,  # ME4/1, ME4/2
   0, 1, 2, 3, 4, 5, 6, 7, 8,  # neigh
  #
   9, 9, 9,10,10,10,11,11,11,  # GE1/1 sub 1, RE1/2 sub 1, RE1/3 sub 1
   9, 9, 9,10,10,10,11,11,11,  # GE1/1 sub 2, RE1/2 sub 2, RE1/3 sub 2
  12,12,12,13,13,13,13,13,13,  # GE2/1, RE2/2
  14,14,14,15,15,15,15,15,15,  # RE3/1, RE3/2
  16,16,16,17,17,17,17,17,17,  # RE4/1, RE4/2
   9,10,11,12,13,14,15,16,17,  # neigh
  #
  18,18,18,18,18,18,18,        # ME0
], dtype=np.int32)

# Assign EMTF zones
def find_emtf_zones_lut():
  default_value = -99
  lut = np.full((num_emtf_hosts, num_emtf_zones, 2), default_value, dtype=np.int32)  # (host,zone) -> (min_theta,max_theta)
  lut[0,0] = 4,26    # ME1/1
  lut[3,0] = 4,25    # ME2/1
  lut[5,0] = 4,25    # ME3/1
  lut[7,0] = 4,25    # ME4/1
  lut[9,0] = 17,26   # GE1/1
  lut[12,0] = 4,25   # GE2/1
  lut[14,0] = 4,25   # RE3/1
  lut[16,0] = 4,25   # RE4/1
  lut[18,0] = 4,23   # ME0
  #
  lut[0,1] = 24,53   # ME1/1
  lut[1,1] = 46,54   # ME1/2
  lut[3,1] = 23,49   # ME2/1
  lut[5,1] = 23,40   # ME3/1
  lut[6,1] = 44,54   # ME3/2
  lut[7,1] = 23,35   # ME4/1
  lut[8,1] = 38,54   # ME4/2
  lut[9,1] = 24,52   # GE1/1
  lut[10,1] = 52,56  # RE1/2
  lut[12,1] = 23,46  # GE2/1
  lut[14,1] = 23,36  # RE3/1
  lut[15,1] = 40,52  # RE3/2
  lut[16,1] = 23,31  # RE4/1
  lut[17,1] = 36,52  # RE4/2
  #
  lut[1,2] = 52,88   # ME1/2
  lut[4,2] = 53,88   # ME2/2
  lut[6,2] = 51,88   # ME3/2
  lut[8,2] = 51,88   # ME4/2
  lut[10,2] = 52,84  # RE1/2
  lut[13,2] = 56,88  # RE2/2
  lut[15,2] = 48,84  # RE3/2
  lut[17,2] = 52,84  # RE4/2
  return lut

def find_emtf_zones_initializer():
  lut = find_emtf_zones_lut()

  def lookup(emtf_host, emtf_theta):
    emtf_host = np.asarray(emtf_host)
    emtf_theta = np.asarray(emtf_theta)
    bounds = np.take(lut, emtf_host, axis=0)
    # Create a boolean array representing the bits in a uint8 array. Set the last 3 bits.
    # Then, pack into a uint8 array
    if emtf_host.ndim == 0:
      result = np.zeros(8, dtype=np.bool)
      result[-3:] = (bounds[..., 0] <= emtf_theta) & (emtf_theta <= bounds[..., 1])
      result = np.packbits(result)
    else:
      result = np.zeros(emtf_host.shape + (8,), dtype=np.bool)
      result[..., -3:] = (bounds[..., 0] <= emtf_theta[:, np.newaxis]) & (emtf_theta[:, np.newaxis] <= bounds[..., 1])
      result = np.packbits(result)
    return result
  return lookup

# The initializer will instantiate the lookup table
find_emtf_zones = find_emtf_zones_initializer()

# Assign EMTF timezones
def find_emtf_timezones_lut():
  default_value = -99
  lut = np.full((num_emtf_hosts, num_emtf_timezones, 2), default_value, dtype=np.int32)  # (host,timezone) -> (min_bx,max_bx)
  lut[0,0] = -1,0  # ME1/1
  lut[1,0] = -1,0  # ME1/2
  lut[2,0] = -1,0  # ME1/3
  lut[3,0] = -1,0  # ME2/1
  lut[4,0] = -1,0  # ME2/2
  lut[5,0] = -1,0  # ME3/1
  lut[6,0] = -1,0  # ME3/2
  lut[7,0] = -1,0  # ME4/1
  lut[8,0] = -1,0  # ME4/2
  lut[9,0] = 0,0   # GE1/1
  lut[10,0] = 0,0  # RE1/2
  lut[11,0] = 0,0  # RE1/3
  lut[12,0] = 0,0  # GE2/1
  lut[13,0] = 0,0  # RE2/2
  lut[14,0] = 0,0  # RE3/1
  lut[15,0] = 0,0  # RE3/2
  lut[16,0] = 0,0  # RE4/1
  lut[17,0] = 0,0  # RE4/2
  lut[18,0] = 0,0  # ME0
  #
  lut[:, 1] = lut[:, 0] - 1  # timezone 1 = timezone 0 - 1
  lut[:, 2] = lut[:, 1] - 1  # timezone 2 = timezone 1 - 1
  return lut

def find_emtf_timezones_initializer():
  lut = find_emtf_timezones_lut()

  def lookup(emtf_host, bx):
    emtf_host = np.asarray(emtf_host)
    bx = np.asarray(bx)
    bounds = np.take(lut, emtf_host, axis=0)
    # Create a boolean array representing the bits in a uint8 array. Set the last 3 bits.
    # Then, pack into a uint8 array
    if emtf_host.ndim == 0:
      result = np.zeros(8, dtype=np.bool)
      result[-3:] = (bounds[..., 0] <= bx) & (bx <= bounds[..., 1])
      result = np.packbits(result)
    else:
      result = np.zeros(emtf_host.shape + (8,), dtype=np.bool)
      result[..., -3:] = (bounds[..., 0] <= bx[:, np.newaxis]) & (bx[:, np.newaxis] <= bounds[..., 1])
      result = np.packbits(result)
    return result
  return lookup

# The initializer will instantiate the lookup table
find_emtf_timezones = find_emtf_timezones_initializer()

# Encode EMTF zone image row
def find_emtf_img_row_lut():
  # From closest to furthest:
  # ME0
  # GE1/1, ME1/1, ME1/2, RE1,
  # GE2/1, RE2, ME2,
  # ME3, RE3,
  # ME4, RE4,
  default_value = -99
  lut = np.full((num_emtf_hosts, num_emtf_zones), default_value, dtype=np.int32)  # (host,zone) -> row
  lut[0,0] = 2   # ME1/1
  lut[3,0] = 4   # ME2/1
  lut[5,0] = 5   # ME3/1
  lut[7,0] = 7   # ME4/1
  lut[9,0] = 1   # GE1/1
  lut[12,0] = 3  # GE2/1
  lut[14,0] = 6  # RE3/1
  lut[16,0] = 7  # RE4/1
  lut[18,0] = 0  # ME0
  #
  lut[0,1] = 1   # ME1/1
  lut[1,1] = 2   # ME1/2
  lut[3,1] = 4   # ME2/1
  lut[5,1] = 5   # ME3/1
  lut[6,1] = 5   # ME3/2
  lut[7,1] = 7   # ME4/1
  lut[8,1] = 7   # ME4/2
  lut[9,1] = 0   # GE1/1
  lut[10,1] = 2  # RE1/2
  lut[12,1] = 3  # GE2/1
  lut[14,1] = 6  # RE3/1
  lut[15,1] = 6  # RE3/2
  lut[16,1] = 7  # RE4/1
  lut[17,1] = 7  # RE4/2
  #
  lut[1,2] = 0   # ME1/2
  lut[4,2] = 3   # ME2/2
  lut[6,2] = 4   # ME3/2
  lut[8,2] = 6   # ME4/2
  lut[10,2] = 1  # RE1/2
  lut[13,2] = 2  # RE2/2
  lut[15,2] = 5  # RE3/2
  lut[17,2] = 7  # RE4/2
  return lut

def find_emtf_img_row_initializer():
  lut = find_emtf_img_row_lut()

  def lookup(emtf_host, zone):
    emtf_host = np.asarray(emtf_host)
    zone = np.asarray(zone)
    if zone.ndim != 0:
      raise ValueError('zone should be a scalar.')
    item = np.take(lut[:, zone], emtf_host)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_img_row = find_emtf_img_row_initializer()

img_row_labels = np.array([
  ['ME0'  , 'GE1/1', 'ME1/1', 'GE2/1', 'ME2/1', 'ME3/1', 'RE3/1', 'ME4/1'],
  ['GE1/1', 'ME1/1', 'ME1/2', 'GE2/1', 'ME2/1', 'ME3/1', 'RE3/1', 'ME4/1'],
  ['ME1/2', 'RE1/2', 'RE2/2', 'ME2/2', 'ME3/2', 'RE3/2', 'ME4/2', 'RE4/2'],
], dtype=np.object)

site_to_img_row_luts = np.array([
  [2, 2, 4, 5, 7, 2, 4, 6, 7, 1, 3, 0],
  [1, 2, 4, 5, 7, 2, 4, 6, 7, 0, 3, 0],
  [0, 0, 3, 4, 6, 1, 2, 5, 7, 0, 3, 0],
], dtype=np.int32)

def find_emtf_img_col(emtf_phi):
  emtf_phi = np.asarray(emtf_phi)
  return (emtf_phi - min_emtf_strip) // coarse_emtf_strip

def find_emtf_img_col_inverse(emtf_img_col):
  emtf_img_col = np.asarray(emtf_img_col)
  return (emtf_img_col * coarse_emtf_strip) + (coarse_emtf_strip // 2) + min_emtf_strip

# ______________________________________________________________________________
def find_emtf_phi(hit):
  #assert (0 <= hit.emtf_phi) and ((hit.type != kDT and hit.emtf_phi < 5000) or (hit.type == kDT))
  emtf_phi = np.int32(hit.emtf_phi)
  return emtf_phi

def find_emtf_theta(hit):
  #assert (hit.emtf_theta > 0)
  emtf_theta = np.int32(hit.emtf_theta)
  if hit.type == kDT:
    # wire -1 means no theta SL
    # quality 0&1 are RPC digis
    if (hit.wire == -1) or (hit.quality < 2):
      if hit.station == 1:
        emtf_theta = 112
      elif hit.station == 2:
        emtf_theta = 122
      elif hit.station == 3:
        emtf_theta = 131
    else:
      pass
  else:  # non-DT
    pass
  return emtf_theta

def find_emtf_bend(hit):
  emtf_bend = np.int32(hit.bend)
  if hit.type == kCSC:
    # Special case for ME1/1a:
    # rescale the bend to the same scale as ME1/1b
    if hit.station == 1 and hit.ring == 4:
      emtf_bend = np.round(emtf_bend.astype(np.float32) * 0.026331/0.014264).astype(np.int32)
      emtf_bend = np.clip(emtf_bend, -32, 31)
    emtf_bend *= hit.endcap
    emtf_bend = np.round(emtf_bend.astype(np.float32) * 0.5001).astype(np.int32)  # from 1/32-strip unit to 1/16-strip unit
    emtf_bend = np.clip(emtf_bend, -16, 15)
  elif hit.type == kME0:
    emtf_bend = np.round(emtf_bend.astype(np.float32) * 0.5001).astype(np.int32)  # from 1/4-strip unit to 1/2-strip unit
    emtf_bend = np.clip(emtf_bend, -64, 63)
  elif hit.type == kDT:
    if hit.quality >= 4:
      emtf_bend = np.clip(emtf_bend, -512, 511)
    else:
      #emtf_bend = np.int32(0)
      emtf_bend = np.clip(emtf_bend, -512, 511)
  else:  # kRPC, kGEM
    emtf_bend = np.int32(0)
  return emtf_bend

def find_emtf_qual(hit):
  emtf_qual = np.int32(hit.quality)
  emtf_qual = np.clip(emtf_qual, 0, 15)
  return emtf_qual

def find_emtf_time(hit):
  emtf_time = np.int32(hit.time)
  emtf_time = np.clip(emtf_time, -8, 7)
  return emtf_time

# ______________________________________________________________________________
# Check whether hit is very legal and very cool
def is_emtf_legit_hit(hit):
  def check_type(hit):
    is_dt = (hit.type == kDT)
    return (not is_dt)
  def check_phi(hit):
    if hasattr(hit, 'emtf_phi'):
      if hit.type == kME0 or hit.type == kDT:
        return hit.emtf_phi > 0
    return True
  def check_theta(hit):
    if hasattr(hit, 'emtf_theta'):
      if hit.type == kME0:
        return hit.emtf_theta > 0
    return True
  return check_type(hit) and check_phi(hit) and check_theta(hit)

# Check whether hit is very legal and very cool for Run 2
def is_emtf_legit_hit_run2(hit):
  def check_type(hit):
    is_csc = (hit.type == kCSC)
    is_rpc_run2 = (hit.type == kRPC) \
        and not ((hit.station == 3 or hit.station == 4) and hit.ring == 1) \
        and not ((hit.station == 1 or hit.station == 2) and hit.ring == 3)
    return (is_csc or is_rpc_run2)
  def check_phi(hit):
    if hasattr(hit, 'emtf_phi'):
      if hit.type == kME0 or hit.type == kDT:
        return hit.emtf_phi > 0
    return True
  def check_theta(hit):
    if hasattr(hit, 'emtf_theta'):
      if hit.type == kME0:
        return hit.emtf_theta > 0
    return True
  return check_type(hit) and check_phi(hit) and check_theta(hit)

# ______________________________________________________________________________
# Find particle zone based on eta
def find_particle_zone(eta):
  ind = hist_digitize_inclusive(np.abs(eta), emtf_eta_bins)
  return (len(emtf_eta_bins)-2) - ind  # ind (0,1,2,3) -> zone (3,2,1,0)
