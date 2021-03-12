"""Utilities for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import six
from six.moves import range, zip, map, filter


# ______________________________________________________________________________
# Enums

kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4

kDEBUG, kINFO, kWARNING, kERROR, kFATAL = 0, 1, 2, 3, 4

# ______________________________________________________________________________
# Configs

def create_assist_module():
  def floatx():
    return 'float32'
  def epsilon():
    return 1e-7
  def image_data_format():
    return 'channels_last'
  def ma_fill_value():
    return 999999

  import types
  assist = types.ModuleType('assist')
  setattr(assist, 'floatx', floatx)
  setattr(assist, 'epsilon', epsilon)
  setattr(assist, 'image_data_format', image_data_format)
  setattr(assist, 'ma_fill_value', ma_fill_value)
  return assist

assist = create_assist_module()

# ______________________________________________________________________________
# Functions

def wrap_phi_rad(x):
  # returns phi in [-pi,pi] rad
  twopi = np.pi*2
  x = x - np.round(x / twopi) * twopi
  return x

def wrap_phi_deg(x):
  # returns phi in [-180.,180] deg
  twopi = 360.
  x = x - np.round(x / twopi) * twopi
  return x

def wrap_theta_rad(x):
  # returns theta in [0,pi/2] rad
  halfpi = np.pi/2
  x = wrap_phi_rad(x)
  x = np.abs(x)
  x = np.where(x >= halfpi, np.pi - x, x)
  return x

def wrap_theta_deg(x):
  # returns theta in [0,90] deg
  halfpi = 90.
  x = wrap_phi_deg(x)
  x = np.abs(x)
  x = np.where(x >= halfpi, 180. - x, x)
  return x

def delta_phi(lhs, rhs):
  # lhs, rhs in radians
  x = wrap_phi_rad(lhs - rhs)
  return x

def delta_theta(lhs, rhs):
  # lhs, rhs in radians
  x = lhs - rhs
  return x

def calc_phi_loc_deg_from_glob(glob, sector):
  # glob in deg, sector [1-6]
  glob = wrap_phi_deg(glob)
  loc = glob - 15. - (60. * (sector-1))
  return loc

def calc_phi_loc_int(glob, sector):
  # glob in deg, sector [1-6]
  loc = calc_phi_loc_deg_from_glob(glob, sector)
  if (loc + 22.) < 0.:
    loc += 360.
  loc = (loc + 22.) * 60.
  phi_int = int(round(loc))
  return phi_int

def calc_phi_loc_deg(bits):
  # bits is an integer
  loc = float(bits) / 60. - 22.
  return loc

def calc_phi_glob_deg(loc, sector):
  # loc in deg, sector [1-6]
  glob = loc + 15. + (60. * (sector-1))
  if glob >= 180.:
    glob -= 360.
  return glob

def calc_theta_int(theta, endcap):
  # theta in deg, endcap [-1,+1]
  if endcap == -1:
    theta = 180. - theta
  theta = (theta - 8.5) * 128. / (45.0-8.5)
  theta_int = int(round(theta))
  return theta_int

def calc_theta_rad_from_eta(eta):
  # returns theta in [0,pi] rad
  theta = np.arctan2(1.0, np.sinh(eta))
  return theta

def calc_theta_deg_from_eta(eta):
  # returns theta in [0,180] deg
  return np.rad2deg(calc_theta_rad_from_eta(eta))

def calc_theta_deg_from_int(theta_int):
  theta_deg = float(theta_int) * (45.0-8.5) / 128. + 8.5
  return theta_deg

def calc_eta_from_theta_rad(theta_rad):
  eta = -1. * np.log(np.tan(theta_rad/2.))
  return eta

def calc_eta_from_theta_deg(theta_deg, endcap):
  # theta in deg, endcap [-1,+1]
  theta_deg = wrap_theta_deg(theta_deg)
  theta_rad = np.deg2rad(theta_deg)
  eta = calc_eta_from_theta_rad(theta_rad)
  if endcap == -1:
    eta = -eta
  return eta

def calc_ns_from_mhz(mhz):
  return 1e3 / mhz

def calc_mhz_from_ns(ns):
  return 1e3 / ns

def calc_quant_scale(num_bits, num_int_bits):
  return 1.0 / (1 << (num_bits - num_int_bits))

def calc_quant_range(num_bits, num_int_bits, narrow_range=False):
  quant_min = 1 if narrow_range else 0
  quant_max = (1 << num_bits) - 1
  zero_point = (quant_max - quant_min + 1) // 2
  zero_point_from_min = quant_min + zero_point
  range_min = quant_min - zero_point_from_min
  range_max = quant_max - zero_point_from_min
  range_min /= (1 << (num_bits - num_int_bits))
  range_max /= (1 << (num_bits - num_int_bits))
  return (range_min, range_max)

def get_trigger_sector(ring, station, chamber):
  result = np.uint32(0)
  if station > 1 and ring > 1:
    # ch 3-8->1, 9-14->2, ... 1,2 -> 6
    result = ((np.uint32(chamber - 3) & 0x7f) // 6) + 1
  elif station == 1:
    # ch 3-8->1, 9-14->2, ... 1,2 -> 6
    result = ((np.uint32(chamber - 3) & 0x7f) // 6) + 1
  else:
    # ch 2-4->1, 5-7->2, ...
    result = ((np.uint32(chamber - 2) & 0x1f) // 3) + 1
  # max sector is 6, some calculations give a value greater than 6 but this is expected.
  result = np.clip(result, 1, 6)
  return result

def get_trigger_subsector(ring, station, chamber):
  # csc_tp_subsector = (tp_station != 1) ? 0 : ((csc_tp_chamber % 6 > 2) ? 1 : 2);
  result = np.uint32(0)
  if station == 1:
    if np.uint32(chamber) % 6 > 2:
      result = result + 1
    else:
      result = result + 2
  return result

def get_trigger_cscid(ring, station, chamber):
  result = np.uint32(0)
  if station == 1:
    result = np.uint32(chamber) % 3 + 1  # 1,2,3
    if ring == 2:
      result = result + 3
    elif ring == 3:
      result = result + 6
  else:
    if ring == 1:
      result = np.uint32(chamber + 1) % 3 + 1  # 1,2,3
    else:
      result = np.uint32(chamber + 3) % 6 + 4  # 4,5,6,7,8,9
  return result

def get_trigger_neighid(ring, station, chamber):
  # neighid is 0 for native chamber, 1 for neighbor chamber
  result = np.uint32(0)
  if station == 1:
    if np.uint32(chamber + 3) % 6 + 1 == 6:
      result = result + 1
  else:
    if ring == 1:
      if np.uint32(chamber + 1) % 3 + 1 == 3:
        result = result + 1
    else:
      if np.uint32(chamber + 3) % 6 + 1 == 6:
        result = result + 1
  return result

def get_trigger_endsec(endcap, sector):
  # endsec is 0-5 in positive endcap, 6-11 in negative endcap
  assert(endcap == 1 or endcap == -1)
  assert(1 <= sector <= 6)
  result = (sector - 1) if endcap == 1 else (sector - 1 + 6)
  return result

def calc_d0_simple(phi, xv, yv):
  d0 = xv * np.sin(phi) - yv * np.cos(phi)
  return d0

def calc_d0(invpt, phi, xv, yv, B=3.811):
  R = -1.0 / (0.003 * B * invpt)            # R = -pT/(0.003 q B)  [cm]
  xc = xv - (R * np.sin(phi))               # xc = xv - R sin(phi)
  yc = yv + (R * np.cos(phi))               # yc = yv + R cos(phi)
  d0 = R - (np.sign(R) * np.hypot(xc, yc))  # d0 = R - sign(R) * sqrt(xc^2 + yc^2)
  return d0

def calc_etastar_from_eta(invpt, eta, phi, x0, y0, z0, zstar=850., zstar_4T=650.):
  # Propagate to station 2 (z = 850 cm), find r and eta of the track
  # (called rstar and etastar).
  # Note: x0, y0, z0 in cm. Assume pT -> inf.
  if eta < 0:
    zstar *= -1
  # Assume a simplified magnetic field where it is 4T (or 3.811T)
  # inside the solenoid and 0T outside (boundary at z = 650 cm)
  if eta < 0:
    zstar_4T *= -1
  B = 3.811
  R = -1.0 / (0.003 * B * invpt)  # R = -pT/(0.003 q B)  [cm], radius of the circle
  cot = np.sinh(eta)              # cot(theta), which is pz/pt
  if np.abs(zstar_4T) < np.abs(zstar):
    arg_term_4T = np.abs((zstar_4T - z0)/cot)                  # with magfield
    sin_term_4T = (2 * R) * np.sin(arg_term_4T/(2 * R))        # with magfield
    cos_term_4T = (2 * R) * (1 - np.cos(arg_term_4T/(2 * R)))  # with magfield
    arg_term_0T = np.abs((zstar - zstar_4T)/cot)               # without magfield
    sin_term_0T = arg_term_0T                                  # without magfield
    cos_term_0T = 0                                            # without magfield
  else:
    # Also need to check for the boundary at r where 4T -> 0T, ignore for now
    arg_term_4T = np.abs((zstar - z0)/cot)                     # with magfield
    sin_term_4T = (2 * R) * np.sin(arg_term_4T/(2 * R))        # with magfield
    cos_term_4T = (2 * R) * (1 - np.cos(arg_term_4T/(2 * R)))  # with magfield
    arg_term_0T = 0                                            # without magfield
    sin_term_0T = 0                                            # without magfield
    cos_term_0T = 0                                            # without magfield
  phistar_4T = phi + arg_term_4T/(2 * R)  # phi at the boundary where 4T -> 0T
  xstar = x0 + np.cos(phi) * sin_term_4T - np.sin(phi) * cos_term_4T + \
      np.cos(phistar_4T) * sin_term_0T - np.sin(phistar_4T) * cos_term_0T
  ystar = y0 + np.sin(phi) * sin_term_4T + np.cos(phi) * cos_term_4T + \
      np.sin(phistar_4T) * sin_term_0T + np.cos(phistar_4T) * cos_term_0T
  rstar = np.hypot(xstar, ystar)
  cotstar = zstar/rstar
  etastar = np.arcsinh(cotstar)
  return etastar

def calc_signed_rvtx(invpt, eta, phi, x0, y0, z0, zstar=850., zstar_4T=650.):
  # Sign is positive if |etastar| <= |eta|, negative otherwise
  etastar = calc_etastar_from_eta(invpt, eta, phi, x0, y0, z0, zstar, zstar_4T)
  rvtx = np.hypot(x0, y0)
  if not (np.abs(etastar) <= np.abs(eta)):
    rvtx *= -1
  return rvtx

def pick_the_median(lst):  # assume sorted list
  middle = 0 if len(lst) == 0 else (len(lst)-1)//2
  return lst[middle]

def pick_the_first(lst):
  return lst[0]

def find_median_of_three(a0, a1, a2, ma_fill_value=999999):
  vld0 = (a0 != ma_fill_value)
  vld1 = (a1 != ma_fill_value)
  vld2 = (a2 != ma_fill_value)
  median = ma_fill_value
  if vld1 and (not vld0 or a0 <= a1) and (vld2 and a1 <= a2):
    median = a1
  elif vld0 and (not vld2 or a2 <= a0) and (vld1 and a0 <= a1):
    median = a0
  elif vld2 and (not vld1 or a1 <= a2) and (vld0 and a2 <= a0):
    median = a2
  elif vld1 and (not vld2 or not a1 <= a2) and (vld0 and not a0 <= a1):
    median = a1
  elif vld0 and (not vld1 or not a0 <= a1) and (vld2 and not a2 <= a0):
    median = a0
  elif vld2 and (not vld0 or not a2 <= a0) and (vld1 and not a1 <= a2):
    median = a2
  elif vld1 and not vld0 and not vld2:
    median = a1
  elif vld0 and not vld2 and not vld1:
    median = a0
  elif vld2 and not vld1 and not vld0:
    median = a2
  return median

def nan_to_num(a, num=0.0, copy=True):
  a = np.array(a, subok=True, copy=copy)
  mask = np.isnan(a)
  np.copyto(a, num, where=mask)
  return a

def save_np_arrays(outfile, outdict):
  from numpy.compat import contextlib_nullcontext
  with contextlib_nullcontext(outfile) as f:
    np.savez_compressed(f, **outdict)

def save_root_histograms(outfile, histograms):
  from rootpy.io import root_open
  with root_open(outfile, 'recreate') as f:
    for (k, v) in six.iteritems(histograms):
      v.Write()

def hist_digitize(x, bins, ma_fill_value=999999):
  """
  Digitize according to how np.histogram() computes the histogram. All but the last
  (righthand-most) bin is half-open i.e. [a, b). The last bin is closed i.e. [a, b].
  Underflow and overflow values return an index set to `ma_fill_value`.

  Examples:
  --------
  >>> hist_digitize(0, [1,2,3,4])
  array(999999)
  >>> hist_digitize(1, [1,2,3,4])
  array(0)
  >>> hist_digitize(1.1, [1,2,3,4])
  array(0)
  >>> hist_digitize(2, [1,2,3,4])
  array(1)
  >>> hist_digitize(3, [1,2,3,4])
  array(2)
  >>> hist_digitize(4, [1,2,3,4])
  array(2)
  >>> hist_digitize(4.1, [1,2,3,4])
  array(999999)
  >>> hist_digitize(5, [1,2,3,4])
  array(999999)
  """
  bin_edges = np.asarray(bins)
  n_bin_edges = bin_edges.size
  if n_bin_edges < 2:
    raise ValueError('`bins` must have size >= 2')
  if bin_edges.ndim != 1:
    raise ValueError('`bins` must be 1d')
  if np.any(bin_edges[:-1] > bin_edges[1:]):
    raise ValueError('`bins` must increase monotonically')
  x = np.asarray(x)
  x = np.ravel(x)
  bin_index = bin_edges.searchsorted(x, side='right')
  bin_index[x == bin_edges[-1]] -= 1
  bin_index[bin_index == 0] = ma_fill_value
  bin_index[bin_index == n_bin_edges] = ma_fill_value
  bin_index[bin_index != ma_fill_value] -= 1
  bin_index = np.squeeze(bin_index)
  return bin_index

def hist_digitize_inclusive(x, bins):
  """
  Digitize according to how np.histogram() computes the histogram. All but the last
  (righthand-most) bin is half-open i.e. [a, b). The last bin is closed i.e. [a, b].
  Underflow values return an index of 0, overflow values return an index of len(bins)-2.

  Examples:
  --------
  >>> hist_digitize_inclusive(0, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(1, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(1.1, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(2, [1,2,3,4])
  array(1)
  >>> hist_digitize_inclusive(3, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(4, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(4.1, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(5, [1,2,3,4])
  array(2)
  """
  bin_edges = np.asarray(bins)
  n_bin_edges = bin_edges.size
  if n_bin_edges < 2:
    raise ValueError('`bins` must have size >= 2')
  if bin_edges.ndim != 1:
    raise ValueError('`bins` must be 1d')
  if np.any(bin_edges[:-1] > bin_edges[1:]):
    raise ValueError('`bins` must increase monotonically')
  x = np.asarray(x)
  x = x.ravel()
  bin_index = bin_edges.searchsorted(x, side='right')
  bin_index[bin_index == n_bin_edges] -= 1
  bin_index[bin_index != 0] -= 1
  bin_index = np.squeeze(bin_index)
  return bin_index

# Copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/generic_utils.py
def make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).

  Args:
      size: Integer, total size of the data to slice into batches.
      batch_size: Integer, batch size.

  Returns:
      A list of tuples of array indices.
  """
  num_batches = int(np.ceil(size / float(batch_size)))
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(0, num_batches)]

# Copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/generic_utils.py
def slice_arrays(arrays, start=None, stop=None):
  """Slice an array or list of arrays.

  This takes an array-like, or a list of
  array-likes, and outputs:
      - arrays[start:stop] if `arrays` is an array-like
      - [x[start:stop] for x in arrays] if `arrays` is a list

  Can also work on list/array of indices: `slice_arrays(x, indices)`

  Args:
      arrays: Single array or list of arrays.
      start: can be an integer index (start index) or a list/array of indices
      stop: integer (stop index); should be None if `start` was a list.

  Returns:
      A slice of the array(s).

  Raises:
      ValueError: If the value of start is a list and stop is not None.
  """
  if arrays is None:
    return [None]
  if isinstance(start, list) and stop is not None:
    raise ValueError('The stop argument has to be None if the value of start '
                     'is a list.')
  elif isinstance(arrays, list):
    if hasattr(start, '__len__'):
      # hdf5 datasets only support list objects as indices
      if hasattr(start, 'shape'):
        start = start.tolist()
      return [None if x is None else x[start] for x in arrays]
    return [
        None if x is None else
        None if not hasattr(x, '__getitem__') else x[start:stop] for x in arrays
    ]
  else:
    if hasattr(start, '__len__'):
      if hasattr(start, 'shape'):
        start = start.tolist()
      return arrays[start]
    if hasattr(start, '__getitem__'):
      return arrays[start:stop]
    return [None]

# Based on
#   https://www.tensorflow.org/guide/ragged_tensor
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_tensor_value.py
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_getitem.py
# Example
#   ragged = RaggedTensorValue(values=np.array([3, 1, 4, 1, 5, 9, 2]), row_splits=np.array([0, 4, 4, 6, 7]))
class RaggedTensorValue(object):
  """Represents the value of a `RaggedTensor`."""

  def __init__(self, values, row_splits):
    """Creates a `RaggedTensorValue`.
    Args:
      values: A numpy array of any type and shape; or a RaggedTensorValue.
      row_splits: A 1-D int32 or int64 numpy array.
    """
    if not (isinstance(row_splits[:1], (np.ndarray, np.generic)) and
            row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int32 or int64 numpy array")
    if not isinstance(values[:1], (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    if (isinstance(values, RaggedTensorValue) and
        row_splits.dtype != values.row_splits.dtype):
      raise ValueError("row_splits and values.row_splits must have "
                       "the same dtype")
    self._values = values
    self._row_splits = row_splits

  row_splits = property(
      lambda self: self._row_splits,
      doc="""The split indices for the ragged tensor value.""")
  values = property(
      lambda self: self._values,
      doc="""The concatenated values for all rows in this tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")

  row_lengths = property(
      lambda self: self._row_splits[1:] - self._row_splits[:-1],
      doc="""The lengths of the rows in this ragged tensor value.""")
  nrows = property(
      lambda self: self._row_splits.shape[0] - 1,
      doc="""The number of rows in this ragged tensor value.""")

  @property
  def shape(self):
    """A tuple indicating the shape of this RaggedTensorValue."""
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]

  def __repr__(self):
    return "RaggedTensorValue(values=%r, row_splits=%r)" % (
        self._values, self._row_splits)

  def __len__(self):
    return len(self.row_splits[:-1])

  def __getitem__(self, row_key):
    if isinstance(row_key, slice):
      # Use row_key to slice the starts & limits.
      new_starts = self.row_splits[:-1][row_key]
      new_limits = self.row_splits[1:][row_key]
      zero_pad = np.arange(1, dtype=self.row_splits.dtype)

      # If there's no slice step, then we can just select a single continuous
      # span of `ragged.values(rt_input)`.
      if row_key.step is None or row_key.step == 1:
        # Construct the new splits.  If new_starts and new_limits are empty,
        # then this reduces to [0].  Otherwise, this reduces to:
        #   concat([[new_starts[0]], new_limits])
        new_splits = np.concatenate(
            [zero_pad[new_starts.size:], new_starts[:1], new_limits],
            axis=0)
        values_start = new_splits[0]
        values_limit = new_splits[-1]
        return RaggedTensorValue(self.values[values_start:values_limit], new_splits - values_start)
      else:
        raise ValueError("slicing with slice step is not supported")

    starts = self.row_splits[:-1]
    limits = self.row_splits[1:]
    row = self.values[starts[row_key]:limits[row_key]]
    return row

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def to_list(self):
    """Returns this ragged tensor value as a nested Python list."""
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(self.nrows)
    ]

  def to_array(self):
    """Returns this ragged tensor value as a nested Numpy array."""
    arr = np.empty((self.nrows,), dtype=np.object)
    for i in range(self.nrows):
      arr[i] = self._values[self._row_splits[i]:self._row_splits[i + 1]]
    return arr

# Based on
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_factory_ops.py
def create_ragged_array(pylist):
  """Construct a constant RaggedTensorValue from a nested list."""

  # Ragged rank for returned value
  ragged_rank = 1

  # Build the splits for each ragged rank, and concatenate the inner values
  # into a single list.
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values

  values = np.asarray(values)
  for row_splits in reversed(nested_splits):
    row_splits = np.asarray(row_splits, dtype=np.int32)
    values = RaggedTensorValue(values, row_splits)
  return values

def ragged_stack(tup):
  if not np.all([x.values.ndim > 1 for x in tup]):
    raise TypeError("ragged values must be at least 2D")

  tup_values = [x.values for x in tup]
  tup_row_splits = [x.row_splits for x in tup]

  new_values = np.vstack(tup_values)
  new_row_splits = [0]
  for row_splits in tup_row_splits:
    # Ignore the first entry in row_splits, as the first entry is always zero.
    # Increment all the entries in row_splits by the last value in new_row_splits.
    new_row_splits.extend(new_row_splits[-1] + row_splits[1:])

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=np.int32)
  new_values = RaggedTensorValue(new_values, new_row_splits)
  return new_values

def ragged_boolean_mask(ragged, mask):
  if not (isinstance(mask[:1], (np.ndarray, np.generic)) and
          mask.dtype in (np.bool,) and mask.ndim == 1):
    raise TypeError("mask must be a 1D bool numpy array")
  if not isinstance(ragged, (RaggedTensorValue,)):
    raise TypeError("ragged must be a RaggedTensorValue")
  if not (ragged.values.shape[0] == mask.shape[0]):
    raise ValueError("The length of ragged.values must be equal to the length of mask")

  data = ragged.values
  new_values = data[mask]

  new_row_lengths = np.zeros((ragged.nrows,), dtype=ragged.row_splits.dtype)
  for i in range(ragged.nrows):
    new_row_lengths[i] = np.count_nonzero(mask[ragged.row_splits[i]:ragged.row_splits[i + 1]])
  new_row_splits = np.append(0, np.cumsum(new_row_lengths))

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=ragged.row_splits.dtype)
  new_values = RaggedTensorValue(new_values, new_row_splits)
  return new_values

def ragged_row_boolean_mask(ragged, row_mask):
  if not (isinstance(row_mask[:1], (np.ndarray, np.generic)) and
          row_mask.dtype in (np.bool,) and row_mask.ndim == 1):
    raise TypeError("row_mask must be a 1D bool numpy array")
  if not isinstance(ragged, (RaggedTensorValue,)):
    raise TypeError("ragged must be a RaggedTensorValue")
  if not (ragged.nrows == row_mask.shape[0]):
    raise ValueError("The number of rows in ragged must be equal to the length of row_mask")

  data = ragged.values
  mask = np.zeros((ragged.values.shape[0],), dtype=np.bool)
  for i in range(ragged.nrows):
    mask[ragged.row_splits[i]:ragged.row_splits[i + 1]] = row_mask[i]
  new_values = data[mask]

  new_row_lengths = ragged.row_lengths[row_mask]
  new_row_splits = np.append(0, np.cumsum(new_row_lengths))

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=ragged.row_splits.dtype)
  new_values = RaggedTensorValue(new_values, new_row_splits)
  return new_values

def ragged_row_splits_to_segment_ids(row_splits):
  if not (isinstance(row_splits[:1], (np.ndarray, np.generic)) and
          row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
    raise TypeError("row_splits must be a 1D int32 or int64 numpy array")

  row_lengths = row_splits[1:] - row_splits[:-1]
  nrows = row_splits.shape[0] - 1
  indices = np.arange(nrows, dtype=row_splits.dtype)
  segment_ids = np.repeat(indices, repeats=row_lengths)
  return segment_ids

def ragged_segment_ids_to_row_splits(segment_ids, num_segments=None):
  if not (isinstance(segment_ids[:1], (np.ndarray, np.generic)) and
          segment_ids.dtype in (np.int64, np.int32) and segment_ids.ndim == 1):
    raise TypeError("segment_ids must be a 1D int32 or int64 numpy array")
  if num_segments is not None:
    if not (isinstance(num_segments, (np.ndarray, np.generic)) and
            num_segments.dtype in (np.int64, np.int32) and num_segments.ndim == 0):
      raise TypeError("num_segment must be a 0D int32 or int64 numpy array")

  row_lengths = np.bincount(segment_ids, minlength=num_segments)
  row_splits = np.append(0, np.cumsum(row_lengths))
  row_splits = np.asarray(row_splits, dtype=segment_ids.dtype)
  return row_splits

def ragged_range(starts, limits=None, deltas=1, dtype=np.int32):
  if limits is None:
    starts = np.asarray(starts)
    starts, limits = (starts * 0), starts
  else:
    starts, limits = np.asarray(starts), np.asarray(limits)

  deltas = np.asarray(deltas)
  if deltas.ndim == 0:
    nested_range = [np.arange(start, limit, deltas, dtype=dtype) for (start, limit) in zip(starts, limits)]
  else:
    nested_range = [np.arange(start, limit, delta, dtype=dtype) for (start, limit, delta) in zip(starts, limits, deltas)]

  values = np.concatenate(nested_range, axis=0)
  row_lengths = np.array([len(x) for x in nested_range])
  row_splits = np.append(0, np.cumsum(row_lengths))
  row_splits = np.asarray(row_splits, dtype=np.int32)
  return RaggedTensorValue(values, row_splits)

# Based on
#   https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/sparse_tensor.py
# Example
#   sparse = SparseTensorValue(indices=np.array([[0, 0], [1, 2], [2, 3]]), values=np.array([1, 2, 3]), dense_shape=np.array([3, 4]))
class SparseTensorValue(object):
  """Represents the value of a `SparseTensor`."""

  def __init__(self, indices, values, dense_shape):
    """Creates a `SparseTensor`.
    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      dense_shape: A 1-D int64 tensor of shape `[ndims]`.
    """
    if not (isinstance(indices[:1], (np.ndarray, np.generic)) and
            indices.dtype in (np.int64, np.int32) and indices.ndim == 2):
      raise TypeError("indices must be a 2D int32 or int64 numpy array")
    if not (isinstance(values[:1], (np.ndarray, np.generic)) and values.ndim >= 1):
      raise TypeError("values must be a n-D numpy array")
    if not (isinstance(dense_shape[:1], (np.ndarray, np.generic)) and
            dense_shape.dtype in (np.int64, np.int32) and dense_shape.ndim == 1):
      raise TypeError("dense_shape must be a 1D int32 or int64 numpy array")
    if not (indices.shape[0] == values.shape[0]):
      raise TypeError("indices and values must have the same first dim")
    if not (indices.shape[1] + (values.ndim - 1) == dense_shape.shape[0]):
      raise TypeError("indices, values, and dense_shape must have consistent shapes")
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

  indices = property(
      lambda self: self._indices,
      doc="""The indices of non-zero values in the represented dense tensor.""")
  values = property(
      lambda self: self._values,
      doc="""The non-zero values in the represented dense tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")
  dense_shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")
  shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")

  def __repr__(self):
    return "SparseTensorValue(indices=%r, values=%r, dense_shape=%r)" % (
        self._indices, self._values, self._dense_shape)

def dense_to_sparse(dense):
  dense = np.asarray(dense)
  indices = np.argwhere(dense)
  values = dense[dense.nonzero()]
  dense_shape = np.asarray(dense.shape)
  return SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)

def sparse_to_dense(sparse):
  dense = np.zeros(sparse.dense_shape, dtype=sparse.dtype)
  ndims = sparse.indices.shape[1]
  tup = tuple(sparse.indices[:, i] for i in range(ndims))
  dense[tup] = sparse.values
  return dense

def sparse_to_dense_n(sparse, n):
  dense_shape = (n,) + sparse.dense_shape[1:]
  dense = np.zeros(dense_shape, dtype=sparse.dtype)
  for i in range(len(sparse.indices)):
    if sparse.indices[i, 0] >= n:
      break
    tup = tuple(sparse.indices[i])
    dense[tup] = sparse.values[i]
  return dense

# Based on
#   https://www.tensorflow.org/api_docs/python/tf/IndexedSlices
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/indexed_slices.py
# Example
#   indexed = IndexedSlicesValue(indices=np.array([0, 2, 4]), values=np.array([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]), dense_shape=np.array([6, 4]))
class IndexedSlicesValue(object):
  """Represents the value of an `IndexedSlices`."""

  def __init__(self, indices, values, dense_shape):
    """Creates an `IndexedSlices`.
    Args:
      indices: A 1-D integer Tensor with shape [D0].
      values: A Tensor of any dtype with shape [D0, D1, ..., Dn].
      dense_shape: A 1-D int64 tensor of shape [ndims], e.g. [LARGE0, D1, .. , DN] where LARGE0 >> D0
    """
    if not (isinstance(indices[:1], (np.ndarray, np.generic)) and
            indices.dtype in (np.int64, np.int32) and indices.ndim == 1):
      raise TypeError("indices must be a 1D int32 or int64 numpy array")
    if not (isinstance(values[:1], (np.ndarray, np.generic)) and values.ndim >= 1):
      raise TypeError("values must be a n-D numpy array")
    if not (isinstance(dense_shape[:1], (np.ndarray, np.generic)) and
            dense_shape.dtype in (np.int64, np.int32) and dense_shape.ndim == 1):
      raise TypeError("dense_shape must be a 1D int32 or int64 numpy array")
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

  indices = property(
      lambda self: self._indices,
      doc="""The indices of non-zero values in the represented dense tensor.""")
  values = property(
      lambda self: self._values,
      doc="""The non-zero values in the represented dense tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")
  dense_shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")
  shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")

  def __repr__(self):
    return "IndexedSlicesValue(indices=%r, values=%r, dense_shape=%r)" % (
        self._indices, self._values, self._dense_shape)

def dense_to_indexed_slices(dense, indices):
  dense = np.asarray(dense)
  indices = np.asarray(indices)
  values = dense[indices]
  dense_shape = np.asarray(dense.shape)
  return IndexedSlicesValue(indices=indices, values=values, dense_shape=dense_shape)

def indexed_slices_to_dense(indexed):
  dense = np.zeros(indexed.dense_shape, dtype=indexed.dtype)
  dense[indexed.indices] = indexed.values
  return dense
