import numpy as np
np.random.seed(2023)

from itertools import izip
from rootpy.plotting import Hist, Hist2D
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, ShortCol
from rootpy.io import root_open

# ______________________________________________________________________________
# Analyzer

# Enums
kDT, kCSC, kRPC, kGEM, kTT = 0, 1, 2, 3, 20

# Lambdas
deg_to_rad = lambda x: x * np.pi/180.

rad_to_deg = lambda x: x * 180./np.pi

# Functions
def delta_phi(lhs, rhs):  # in radians
  rad = lhs - rhs
  while rad <  -np.pi:  rad += np.pi*2
  while rad >= +np.pi:  rad -= np.pi*2
  return rad

def delta_theta(lhs, rhs):  # in radians
  rad = lhs - rhs
  return rad

def select_by_eta(eta):
  return 1.24 <= abs(eta) < 2.4

def select_by_bx(bx):
  return bx == 0

def select_by_vertex(vx, vy, vz):
  return np.sqrt(vx*vx + vy*vy) < 15. and abs(vz) < 50.

# Book histograms
histograms = {}
histogram2Ds = {}

hname, htitle = "muon_eta_vs_pt", "; 1/p_{T} [1/GeV]; |#eta|"
histogram2Ds[hname] = Hist2D(100, -0.2, 0.2, 65, 1.2, 2.5, name=hname, title=htitle, type='F')


# ______________________________________________________________________________
# Open file
infile = root_open('rateplots_mc_r305310_run2_all.1.root')
tree = infile.ntupler.tree

# Define collections
tree.define_collection(name='hits', prefix='vh_', size='vh_size')
tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
tree.define_collection(name='particles', prefix='vp_', size='vp_size')

# Get number of events
#maxEvents = -1
maxEvents = 100000

# ______________________________________________________________________________
# Loop over events
for ievt, evt in enumerate(tree):
  if maxEvents != -1 and ievt == maxEvents:
    break

  # ____________________________________________________________________________
  # Verbose

  verbose = False

  if verbose:
    if (ievt % 1 == 0):  print("Processing event: {0}".format(ievt))
    # Hits
    for ihit, hit in enumerate(evt.hits):
      print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sector, hit.fr, hit.sim_phi, hit.sim_theta, hit.sim_tp1, hit.sim_tp2))
    # Tracks
    for itrk, trk in enumerate(evt.tracks):
      print(".. trk  {0} {1} {2} {3} {4} {5} {6} {7}".format(itrk, trk.sector, trk.mode, trk.pt, trk.phi, trk.eta, trk.theta, trk.q))
    # Gen particles
    for ipart, part in enumerate(evt.particles):
      print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))
  else:
    if (ievt % 1000 == 0):  print("Processing event: {0}".format(ievt))

  # ____________________________________________________________________________
  # Analysis

  h = histogram2Ds["muon_eta_vs_pt"]
  for ipart, part in enumerate(evt.particles):
    if part.pt > 5.:
      if select_by_eta(part.eta):
        if select_by_bx(part.bx):
          if select_by_vertex(part.vx, part.vy, part.vz):
            h.fill(float(part.q)/part.pt, abs(part.eta))


# ______________________________________________________________________________
# End job

with root_open("histos_rootpy.root", "recreate") as f:
  for k, v in histograms.iteritems():
    v.Write()
  for k, v in histogram2Ds.iteritems():
    v.Write()
