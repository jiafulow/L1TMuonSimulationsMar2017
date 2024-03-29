"""Data preparation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_algos import *
from emtf_ntuples import *


# ______________________________________________________________________________
# Classes

class EMTFSectorRanking(object):
  def __init__(self):
    self.sectors = np.zeros(num_emtf_sectors, dtype=np.int32)

  def reset(self):
    self.sectors.fill(0)

  def add(self, hit):
    emtf_host = find_emtf_host(hit.type, hit.station, hit.ring)
    assert (0 <= emtf_host and emtf_host < num_emtf_hosts)
    valid_flag = np.zeros(8, dtype=np.bool)
    valid_flag[0] = (emtf_host == 18)               # ME0
    valid_flag[1] = (emtf_host == 0)                # ME1/1
    valid_flag[2] = (emtf_host in (1,2))            # ME1/2, ME1/3
    valid_flag[3] = (emtf_host in (3,4))            # ME2/1, ME2/2
    valid_flag[4] = (emtf_host in (5,6))            # ME3/1, ME3/2
    valid_flag[5] = (emtf_host in (7,8))            # ME4/1, ME4/2
    valid_flag[6] = (emtf_host in (9,10,11,12,13))  # GE1/1, RE1/2, RE1/3, GE2/1, RE2/2
    valid_flag[7] = (emtf_host in (14,15,16,17))    # RE3/1, RE3/2, RE4/1, RE4/2
    rank = np.packbits(valid_flag)                  # pack 8 booleans into an uint8
    endsec = get_trigger_endsec(hit.endcap, hit.sector)
    self.sectors[endsec] |= rank

  def get_best_sector(self):
    best_sector = np.argmax(self.sectors)
    best_sector_rank = self.sectors[best_sector]
    return (best_sector, best_sector_rank)

class EMTFChamberCouncil(object):
  def __init__(self, is_sim=False):
    self.chambers = {}
    self.is_sim = is_sim

  def reset(self):
    self.chambers.clear()

  def add(self, hit):
    # Call functions
    emtf_site = find_emtf_site(hit.type, hit.station, hit.ring)
    emtf_host = find_emtf_host(hit.type, hit.station, hit.ring)
    emtf_chamber = find_emtf_chamber(hit.type, hit.station, hit.cscid, hit.subsector, hit.neighbor)
    emtf_segment = 0  # dummy
    zones = find_emtf_zones(emtf_host, hit.emtf_theta)
    timezones = find_emtf_timezones(emtf_host, hit.bx)

    emtf_phi = find_emtf_phi(hit)
    emtf_bend = find_emtf_bend(hit)
    emtf_theta = find_emtf_theta(hit)
    emtf_theta_alt = emtf_theta
    emtf_qual = find_emtf_qual(hit)
    emtf_qual_alt = emtf_qual
    emtf_time = find_emtf_time(hit)

    assert(emtf_site != -99)
    assert(emtf_host != -99)
    assert(emtf_chamber != -99)

    # Assign variables
    hit.emtf_site = emtf_site
    hit.emtf_host = emtf_host
    hit.emtf_chamber = emtf_chamber
    hit.emtf_segment = emtf_segment
    hit.zones = zones
    hit.timezones = timezones

    hit.emtf_phi = emtf_phi
    hit.emtf_bend = emtf_bend
    hit.emtf_theta = emtf_theta
    hit.emtf_theta_alt = emtf_theta_alt
    hit.emtf_qual = emtf_qual
    hit.emtf_qual_alt = emtf_qual_alt
    hit.emtf_time = emtf_time

    try:
      hit.detlayer = hit.layer
    except:
      hit.detlayer = 0

    # Add hit according to (bx,chamber)
    k = (hit.bx, hit.emtf_chamber)
    if k not in self.chambers:
      self.chambers[k] = []
    self.chambers[k].append(hit)

  def _to_numpy(self, hits):
    # Send 18 variables
    getter = lambda hit: [hit.emtf_site, hit.emtf_host, hit.emtf_chamber, hit.emtf_segment, hit.zones, hit.timezones,] + \
        [hit.emtf_phi, hit.emtf_bend, hit.emtf_theta, hit.emtf_theta_alt, hit.emtf_qual, hit.emtf_qual_alt, hit.emtf_time,] + \
        [hit.strip, hit.wire, hit.fr, hit.detlayer, hit.bx,]
    arr = np.array([getter(hit) for hit in hits], dtype=np.int32)
    return arr

  def _get_hits_from_chambers_sim(self):
    hits = []
    sorted_keys = sorted(self.chambers.keys())

    for k in sorted_keys:
      (bx, emtf_chamber) = k
      tmp_hits = self.chambers[k]

      # If more than one hit, sort by layer number.
      # For RPC and GEM, keep the first one; For CSC, ME0 and DT, keep the median
      ind = 0
      if len(tmp_hits) > 1:
        tmp_hits.sort(key=lambda hit: hit.detlayer)
        if tmp_hits[0].type == kRPC or tmp_hits[0].type == kGEM:
          ind = 0
        else:
          ind = (len(tmp_hits)-1)//2

      hit = tmp_hits[ind]
      hits.append(hit)
    return hits

  def _get_hits_from_chambers(self):
    hits = []
    sorted_keys = sorted(self.chambers.keys())

    for k in sorted_keys:
      (bx, emtf_chamber) = k
      tmp_hits = self.chambers[k]

      # For CSC, remove the "ghosts" and keep only 2 LCTs per chamber
      if emtf_chamber < 54:  # CSC
        assert len(tmp_hits) in (1,2,4)
        emtf_phi_a = np.min([hit.emtf_phi for hit in tmp_hits])
        emtf_phi_b = np.max([hit.emtf_phi for hit in tmp_hits])
        emtf_theta_a = np.min([hit.emtf_theta for hit in tmp_hits])
        emtf_theta_b = np.max([hit.emtf_theta for hit in tmp_hits])
        emtf_segment = 0

        for hit in tmp_hits:
          keep = False
          if (hit.emtf_phi == emtf_phi_a) and (hit.emtf_theta == emtf_theta_a):
            keep = True
            hit.emtf_theta_alt = emtf_theta_b
          elif (hit.emtf_phi == emtf_phi_b) and (hit.emtf_theta == emtf_theta_b):
            keep = True
            hit.emtf_theta_alt = emtf_theta_a

          if keep:
            hit.emtf_segment = emtf_segment
            emtf_segment += 1
            hits.append(hit)

          if keep and (emtf_phi_a == emtf_phi_b):
            break

      else:  # non-CSC
        if emtf_chamber in (108,109,110,111,112,113,114):  # ME0
          #assert len(tmp_hits) <= 20
          if not len(tmp_hits) <= 20:
            print('[WARNING] emtf_chamber {0} has {1} hits'.format(emtf_chamber, len(tmp_hits)))
        elif emtf_chamber in (54,55,56,63,64,65,99):  # GE1/1
          assert len(tmp_hits) <= 8
        elif emtf_chamber in (72,73,74,102):          # GE2/1
          assert len(tmp_hits) <= 8
        elif emtf_chamber in (81,82,83,104):          # RE3/1
          assert len(tmp_hits) <= 4
        elif emtf_chamber in (90,91,92,106):          # RE4/1
          assert len(tmp_hits) <= 4
        elif (75 <= emtf_chamber <= 98) or emtf_chamber in (103,105,107):  # RE2,3,4/2
          assert len(tmp_hits) <= 4
        else:                                         # RE1/2, RE1/3
          assert len(tmp_hits) <= 2
        emtf_segment = 0

        for hit in tmp_hits:
          keep = True
          if keep:
            hit.emtf_segment = emtf_segment
            emtf_segment += 1
            hits.append(hit)
    return hits

  def get_hits(self):
    if self.is_sim:
      hits = self._get_hits_from_chambers_sim()
    else:
      hits = self._get_hits_from_chambers()
    hits_array = self._to_numpy(hits)
    return hits_array


# ______________________________________________________________________________
# Analyses

class _BaseAnalysis(object):
  """Abstract base class"""
  pass

# ______________________________________________________________________________
class SignalAnalysis(_BaseAnalysis):
  """Prepare signal data used for training.

  Description.
  """

  def run(self, algo, signal='prompt'):
    out_part = []
    out_hits = []
    out_simhits = []

    sectors = EMTFSectorRanking()
    chambers = EMTFChamberCouncil()
    chambers_sim = EMTFChamberCouncil(is_sim=True)

    # __________________________________________________________________________
    # Load tree
    if signal == 'prompt':
      tree = load_pgun_batch(jobid)
    elif signal == 'displ':
      tree = load_pgun_displ_batch(jobid)
    else:
      raise RuntimeError('Unexpected signal: {0}'.format(signal))

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      if (ievt % 1000) == 0:
        print('Processing event {0}'.format(ievt))

      # Reset
      sectors.reset()
      chambers.reset()
      chambers_sim.reset()

      # Particles
      try:
        part = evt.particles[0]  # particle gun
      except:
        continue

      # First, determine the best sector

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit):
          if hit.type == kME0:
            # Special case for ME0 as it is a 20-deg chamber in station 1
            hack_me0_hit_chamber(hit)
          sectors.add(hit)

      (best_sector, best_sector_rank) = sectors.get_best_sector()

      # Second, fill the chambers with trigger primitives

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit) and get_trigger_endsec(hit.endcap, hit.sector) == best_sector:
          chambers.add(hit)

      # Third, fill the chambers with sim hits

      # Sim hits
      for isimhit, simhit in enumerate(evt.simhits):
        simhit.endcap = +1 if simhit.z >= 0 else -1
        if is_emtf_legit_hit(simhit):
          if simhit.type == kME0:
            # Special case for ME0 as it is a 20-deg chamber in station 1
            hack_me0_hit_chamber(simhit)

          simhit.emtf_phi = calc_phi_loc_int(np.rad2deg(simhit.phi), (best_sector%6) + 1)
          simhit.emtf_theta = calc_theta_int(np.rad2deg(simhit.theta), 1 if best_sector < 6 else -1)

          simhit.sector = get_trigger_sector(simhit.ring, simhit.station, simhit.chamber)
          simhit.subsector = get_trigger_subsector(simhit.ring, simhit.station, simhit.chamber)
          simhit.cscid = get_trigger_cscid(simhit.ring, simhit.station, simhit.chamber)
          simhit.neighid = get_trigger_neighid(simhit.ring, simhit.station, simhit.chamber)
          simhit.bx = simhit.bend = simhit.quality = simhit.time = simhit.strip = simhit.wire = simhit.fr = 0  # dummy

          # If neighbor, share simhit with the neighbor sector
          get_prev_sector = lambda sector: (sector - 1) if sector != 1 else (sector - 1 + 6)
          get_next_sector = lambda sector: (sector + 1) if sector != 6 else (sector + 1 - 6)
          if get_trigger_endsec(simhit.endcap, simhit.sector) == best_sector:
            simhit.neighbor = 0
            chambers_sim.add(simhit)
          elif get_trigger_endsec(simhit.endcap, get_next_sector(simhit.sector)) == best_sector:
            if simhit.neighid == 1:
              simhit.neighbor = 1
              simhit.sector = get_next_sector(simhit.sector)
              chambers_sim.add(simhit)
          elif get_trigger_endsec(simhit.endcap, get_prev_sector(simhit.sector)) == best_sector:
            if simhit.type == kME0 and simhit.emtf_phi >= ((55 + 22) * 60):
              # Special case for ME0 as there is a 5 deg shift.
              # The CSC chamber 1 starts at -5 deg, but the ME0 chamber 1 starts at -10 deg.
              simhit.neighbor = 0
              simhit.sector = get_prev_sector(simhit.sector)
              chambers_sim.add(simhit)

      # Fourth, extract the particle and hits
      def get_part_info():
        etastar = calc_etastar_from_eta(part.invpt, part.eta, part.phi, part.vx, part.vy, part.vz)
        part_zone = find_particle_zone(etastar)
        part_info = [part.invpt, part.eta, part.phi, part.vx, part.vy, part.vz, part.d0, best_sector, part_zone]
        part_info = np.array(part_info, dtype=np.float32)
        return part_info

      ievt_part = get_part_info()
      ievt_hits = chambers.get_hits()
      ievt_simhits = chambers_sim.get_hits()

      # Fifth, check for at least 2 stations (using sim hits)
      require_two_stations = True

      if require_two_stations:
        _stations = [hit.station for tmp_hits in six.itervalues(chambers_sim.chambers) for hit in tmp_hits]
        _types = [hit.type for tmp_hits in six.itervalues(chambers_sim.chambers) for hit in tmp_hits]
        min_station = np.min(_stations) if len(_stations) else 5
        max_station = np.max(_stations) if len(_stations) else 0
        both_me0_me1 = (kME0 in _types) and (kCSC in _types)
        keep = (min_station <= 1 and max_station >= 2) or (min_station == 2 and max_station >= 3) or (max_station == 1 and both_me0_me1)
        if not keep:
          ievt_hits = np.array([], dtype=np.int32)
          ievt_simhits = np.array([], dtype=np.int32)

      # Finally, add to output
      out_part.append(ievt_part)
      out_hits.append(ievt_hits)
      out_simhits.append(ievt_simhits)

      # Debug
      if verbosity >= kINFO:
        print('Processing event {0}'.format(ievt))
        print('.. part {0} {1} {2} {3} {4} {5}'.format(0, part.pt, part.eta, part.phi, part.invpt, part.d0))
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, get_trigger_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          if (hit.type == kCSC) and (hit_sim_tp != hit.sim_tp2):
            hit_sim_tp = -1
          print('.. hit {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(ihit, hit_id, hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit_sim_tp, hit.strip, hit.wire))
        for isimhit, simhit in enumerate(evt.simhits):
          try:
            simhit_id = (simhit.type, simhit.station, simhit.ring, get_trigger_endsec(simhit.endcap, simhit.sector), simhit.fr, simhit.bx)
            print('.. simhit {0} {1} {2} {3}'.format(isimhit, simhit_id, simhit.phi, simhit.theta))
          except:
            pass
        print('best sector: {0} rank: {1}'.format(best_sector, best_sector_rank))
        with np.printoptions(linewidth=100, threshold=1000):
          print('hits:')
          print(ievt_hits)
          print('simhits:')
          print(ievt_simhits)

    # End loop over events

    # __________________________________________________________________________
    # Output
    outfile = 'signal.npz'
    if use_condor:
      outfile = outfile[:-4] + ('_%i.npz' % jobid)
    out_part = np.asarray(out_part)
    out_hits = create_ragged_array(out_hits)
    out_simhits = create_ragged_array(out_simhits)
    print('[INFO] out_part: {0} out_hits: {1} out_simhits: {2}'.format(out_part.shape, out_hits.shape, out_simhits.shape))
    outdict = {
      'out_part': out_part,
      'out_hits_values': out_hits.values,
      'out_hits_row_splits': out_hits.row_splits,
      'out_simhits_values': out_simhits.values,
      'out_simhits_row_splits': out_simhits.row_splits,
    }
    save_np_arrays(outfile, outdict)
    return

# ______________________________________________________________________________
class BkgndAnalysis(_BaseAnalysis):
  """Prepare background data used for training.

  Description.
  """

  def run(self, algo):
    out_aux = []
    out_hits = []

    sector_chambers = [EMTFChamberCouncil() for sector in range(num_emtf_sectors)]

    # __________________________________________________________________________
    # Load tree
    tree = load_mixing_batch(jobid)

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      if (ievt % 100) == 0:
        print('Processing event {0}'.format(ievt))

      # Reset
      for sector in range(num_emtf_sectors):
        sector_chambers[sector].reset()

      # First, apply event veto

      # Particles
      veto = False
      for ipart, part in enumerate(evt.particles):
        if (part.bx == 0) and (part.pt > 14.) and (find_particle_zone(part.eta) in (0,1,2,3)):
          veto = True
          break
      if veto:
        continue

      # Second, fill the zones with trigger primitives

      # Trigger primitives
      for sector in range(num_emtf_sectors):
        for ihit, hit in enumerate(evt.hits):
          if is_emtf_legit_hit(hit) and get_trigger_endsec(hit.endcap, hit.sector) == sector:
            sector_chambers[sector].add(hit)

      # Finally, extract the particle and hits. Add them to output.
      def get_aux_info():
        aux_info = [jobid, ievt, sector]
        aux_info = np.array(aux_info, dtype=np.int32)
        return aux_info

      for sector in range(num_emtf_sectors):
        ievt_aux = get_aux_info()
        ievt_hits = sector_chambers[sector].get_hits()
        out_aux.append(ievt_aux)
        out_hits.append(ievt_hits)

      # Debug
      if verbosity >= kINFO:
        print('Processing event {0}'.format(ievt))
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, get_trigger_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          if (hit.type == kCSC) and (hit_sim_tp != hit.sim_tp2):
            hit_sim_tp = -1
          print('.. hit {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(ihit, hit_id, hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit_sim_tp, hit.strip, hit.wire))
        with np.printoptions(linewidth=100, threshold=1000):
          for sector in range(num_emtf_sectors):
            ievt_hits = sector_chambers[sector].get_hits()
            print('sector {0} hits:'.format(sector))
            print(ievt_hits)

    # End loop over events

    # __________________________________________________________________________
    # Output
    outfile = 'bkgnd.npz'
    if use_condor:
      outfile = outfile[:-4] + ('_%i.npz' % jobid)
    out_aux = np.asarray(out_aux)
    out_hits = create_ragged_array(out_hits)
    print('[INFO] out_hits: {0}'.format(out_hits.shape))
    outdict = {
      'out_aux': out_aux,
      'out_hits_values': out_hits.values,
      'out_hits_row_splits': out_hits.row_splits,
    }
    save_np_arrays(outfile, outdict)
    return


# ______________________________________________________________________________
# Main

import os, sys, datetime

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'

# Analysis mode (pick one)
analysis = 'signal'
#analysis = 'signal_displ'
#analysis = 'bkgnd'

# Job id (pick an integer)
jobid = 0

# Max num of events (-1 means all events)
maxevents = 100

# Verbosity
verbosity = 1

# Condor or not
# if 'CONDOR_EXEC' is defined, overwrite the 3 arguments (algo, analysis, jobid)
use_condor = ('CONDOR_EXEC' in os.environ)
if use_condor:
  nargs = 3
  if len(sys.argv) != (nargs + 1):
    raise RuntimeError('Expect num of arguments: {}'.format(nargs))
  os.environ['ROOTPY_GRIDMODE'] = 'true'
  algo = sys.argv[1]
  analysis = sys.argv[2]
  jobid = int(sys.argv[3])
  maxevents = -1
  verbosity = 0

# Decorator
def app_decorator(fn):
  def wrapper(*args, **kwargs):
    # Begin
    start_time = datetime.datetime.now()
    print('[INFO] Current time    : {}'.format(start_time))
    print('[INFO] Using cmssw     : {}'.format(os.environ['CMSSW_VERSION']))
    print('[INFO] Using condor    : {}'.format(use_condor))
    print('[INFO] Using algo      : {}'.format(algo))
    print('[INFO] Using analysis  : {}'.format(analysis))
    print('[INFO] Using jobid     : {}'.format(jobid))
    print('[INFO] Using maxevents : {}'.format(maxevents))
    # Run
    fn(*args, **kwargs)
    # End
    stop_time = datetime.datetime.now()
    print('[INFO] Elapsed time    : {}'.format(stop_time - start_time))
    return
  return wrapper

# App
@app_decorator
def app():
  # Select analysis
  if analysis == 'signal':
    myapp = SignalAnalysis()
    myargs = dict(algo=algo, signal='prompt')
  elif analysis == 'signal_displ':
    myapp = SignalAnalysis()
    myargs = dict(algo=algo, signal='displ')

  elif analysis == 'bkgnd':
    myapp = BkgndAnalysis()
    myargs = dict(algo=algo)

  else:
    raise RuntimeError('Cannot recognize analysis: {}'.format(analysis))

  # Run analysis
  myapp.run(**myargs)
  return

# Finally
if __name__ == '__main__':
  app()
