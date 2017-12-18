#!/usr/bin/env python

from __future__ import print_function, division
from util import nop
from copy import copy
from collections import Sequence

import numpy as np
import math

import sys
if sys.version_info[0] < 3:
    range = xrange


def features():
  return (f for f in length_feature)


length_one = lambda _: 1
length_two = lambda _: 2
length_cols = lambda game: game.own.field.width
length_rows = lambda game: game.own.field.height
length_cols_two = lambda game: game.own.field.width * 2

length_feature = {
  'points': length_one,
  'combo': length_one,
  'holes': length_cols,
  'holes2': length_cols,
  'ceiling': length_one,
  'ceiling2': length_one,
  'landing': length_one,
  'landing_base': length_one,
  'row_trans': length_rows,
  'col_trans': length_cols,
  'heights': length_cols,
  'wells': length_cols,
  'wells2': length_cols,
  'lines': length_one,
  'hole_depths': length_one,
  'cap': length_one,
  'tspin': length_one,
  'wide2': length_two,
  'holes+': length_cols_two
}

def length(feature, game):
  try:
    if isinstance(feature, tuple):
      return 1 #np.product([length(x, game) for x in feature])
    return length_feature[feature](game)
  except KeyError:
    return 1

def broadcast(feature, game, vector):
  feature_len = length(feature, game)
  if isinstance(vector, np.ndarray):
    vector = tuple(x for x in vector)
  is_seq = isinstance(vector, Sequence)
  if feature_len == 1:
    return vector[0] if is_seq else vector
  if not is_seq:
    vector = [vector]
  repeat = int(math.ceil(feature_len / len(vector)))
  return (vector * repeat)[:feature_len]


def calculate_points(game):
  return game.own.points

def calculate_combo(game):
  return game.own.combo

def count_holes(col):
  under = False
  holes = 0
  for x in col:
    if under:
      holes += not x
    else:
      under = under or x
  return holes

def calculate_holes(game, slice = slice(None,None,None)):
  ceiling, height = game.own.field.ceiling, game.own.field.real_height
  arr = game.own.field.cells[ceiling:height,slice] #
  return np.apply_along_axis(count_holes, 0, arr)

def calculate_holes2(game, slice = slice(None,None,None)):
  holes = calculate_holes(game)
  return holes * holes

def calculate_ceiling(game):
  return game.own.field.height - game.own.field.ceiling + game.height_penalty

def calculate_ceiling2(game):
  ceiling = calculate_ceiling(game)
  return ceiling * ceiling

def calculate_ceiling3(game):
  ceiling = calculate_ceiling(game)
  return ceiling * ceiling * ceiling

def calculate_landing(game):
  return 0

def calculate_landing_base(game):
  return 0

def calculate_transitions(arr, axis):
  diff = np.diff(arr, axis = axis)
  return np.sum(np.abs(diff, diff), axis = axis)

def calculate_row_transitions(game, slice = slice(None,None,None)):
  arr = game.own.field.cells[slice,:]
  return calculate_transitions(arr, 1)

def calculate_col_transitions(game, slice = slice(None,None,None)):
  ceiling, height = game.own.field.ceiling, game.own.field.real_height
  arr = game.own.field.cells[ceiling:height,slice] #
  return calculate_transitions(arr, 0)

def calculate_heights(game, slice = slice(None,None,None)):
  return game.own.field.heights[slice]

  # if game.own.field.height == game.own.field.ceiling:
  #   return np.zeros((game.own.field.width,), dtype=np.int8)[slice]
  # arr = game.own.field.cells[game.own.field.ceiling:,slice]
  # top = ~arr[0,:]
  # heights = np.argmax(arr, axis=0)
  # zeros = np.where(heights == 0)
  # heights[zeros] = top[zeros] * arr.shape[0]
  # # heights += game.own.field.ceiling
  # return game.own.field.height - game.own.field.ceiling - heights

def calculate_wells(game, heights = None):
  if heights is None:
    heights = calculate_heights(game)
  length = heights.shape[0]
  wells = np.zeros((length,), dtype = np.int8)
  wells[0] = max(0, heights[1] - heights[0])
  for i in range(1, length - 1):
    wells[i] = max(0, min(heights[i-1] - heights[i], heights[i+1] - heights[i]))
  wells[-1] = max(0, heights[-2] - heights[-1])
  wells += (wells > 4) * (wells - 4)
  return wells

def calculate_wells2(game, slice = slice(None,None,None)):
  wells = calculate_wells(game)
  return wells * wells

def calculate_lines(game):
  return 0

def calculate_capped_ceiling(game):
  return max(14, calculate_ceiling(game))

def calculate_capped_ceiling2(game):
  cap = calculate_capped_ceiling(game)
  return cap * cap

def calculate_capped_2ceiling(game):
  cap = calculate_capped_ceiling(game)
  return 2**cap

def count_hole_depths(col):
  depth = 0
  holes = 0
  for x in col:
    if depth:
      holes += not x
      depth += 1
    else:
      depth += x
  return holes + depth

def calculate_hole_depths(game, slice = slice(None,None,None)):
  ceiling, height = game.own.field.ceiling, game.own.field.real_height
  arr = game.own.field.cells[ceiling:height,slice] #
  return np.sum(np.apply_along_axis(count_holes, 0, arr))

tspin_cache = None

def tspin_indexes(height, width):
  if tspin_cache:
    yield tspin_cache
  for (row, col) in np.ndindex((height, width)):
    yield row, col

def tspin_height_diff(game, col):
  heights = game.own.field.heights
  w = game.own.field.width
  left_begin = col
  left_end = col - 1
  right_begin = col + 2
  right_end = col + 3
  return (left_end >= 0 and abs(heights[left_begin] - heights[left_end]) < 2 or
          right_end < w and abs(heights[right_begin] - heights[right_end] < 2))

def calculate_tspin(game):
  floor = game.own.field.floor
  ceiling = max(min(floor - 3, game.own.field.ceiling), 12)
  arr = game.own.field.cells[ceiling:floor,:] #
  shape = arr.shape
  height = shape[0] - 2
  width = shape[1] - 2
  if height <= 0 or width <= 0:
    return 0
  best_progress = 0
  for (row, col) in tspin_indexes(height, width):
    row = height - 1 - row
    col = width - 1 - col
    subarr = arr[row:row+3, col:col+3]
    if subarr[2,1] or subarr[0,1] or subarr[1,:].any():
      continue
    if subarr[0,0] and subarr[0,2]:
      continue
    progress = (int(subarr[0,0]) + int(subarr[0,2]) +
                int(subarr[2,0]) + int(subarr[2,2]) )
    if progress > best_progress:
      best_progress = progress
    if best_progress >= 3:
      completion = (np.sum(arr[row+1:row+3,:]) / 16)
      best_progress += completion * completion
      tspin_cache = (row, col)
      break
  height_penalty = (row / (height - 1)) if height > 1 else 0
  col_penalty = abs((width - 1) / 2 - col) * 2 / (width - 1)
  # support_penalty = best_progress <= 2 and not tspin_height_diff(game, col)
  # line_penalty = 1 - np.sum(arr[row+1:row+3,:]) / 16
  return best_progress - height_penalty * 0.9 - col_penalty * 0.05

def calculate_wide2(game):
  return calculate_heights(game)[-2:]

def count_holes_plus(col):
  min_depth = 0
  holes = 0
  for x in col:
    if min_depth:
      holes += not x
      if not holes:
        min_depth += 1
    else:
      min_depth += x
  return holes, (holes > 0) * (min_depth - 1)

def calculate_holes_plus(game, slice = slice(None,None,None)):
  ceiling, height = game.own.field.ceiling, game.own.field.real_height
  arr = game.own.field.cells[ceiling:height,slice] #
  holes_plus = np.apply_along_axis(count_holes_plus, 0, arr)
  return np.ravel(holes_plus, order='F')

calculate_feature = {
  'points': calculate_points,
  'combo': calculate_combo,
  'holes': calculate_holes,
  'holes2': calculate_holes2,
  'ceiling': calculate_ceiling,
  'ceiling2': calculate_ceiling2,
  'ceiling3': calculate_ceiling3,
  'landing': calculate_landing,
  'landing_base': calculate_landing_base,
  'row_trans': calculate_row_transitions,
  'col_trans': calculate_col_transitions,
  'heights': calculate_heights,
  'wells': calculate_wells,
  'wells2': calculate_wells2,
  'lines': calculate_lines,
  'cap': calculate_capped_ceiling,
  'hole_depths': calculate_hole_depths,
  'tspin': calculate_tspin,
  'wide2': calculate_wide2,
  'holes+': calculate_holes_plus,
  'cap2': calculate_capped_ceiling2,
  '2cap': calculate_capped_2ceiling
}

def calculate(feature, game):
  try:
    return calculate_feature[feature](game)
  except KeyError:
    return 0


def update_holes(holes, game, placement, filled_rows):
  if placement is None:
    return holes
  if filled_rows:
    return calculate_holes(game)
  else:
    col_slice = slice(*placement.col_bounds)
    holes = copy(holes)
    holes[col_slice] = calculate_holes(game, col_slice)
    return holes

def update_landing(landing, game, placement, filled_rows):
  if placement is None:
    return landing + max(0, game.own.field.ceiling - game.piece.max_height)
  row = placement.row_bounds[0] #(placement.row_bounds[0] + placement.row_bounds[1]) / 2
  col = abs(placement.col_bounds[0] + placement.col_bounds[1] - 9) / 9
  return landing + row + col

def update_landing_base(landing, game, placement, filled_rows):
  return landing + placement.row_bounds[1]

def update_lines(lines, game, placement, filled_rows):
  return lines + len(filled_rows)

def update_heights(heights, game, placement, filled_rows):
  if filled_rows:
    return calculate_heights(game)
  else:
    col_slice = slice(*placement.col_bounds)
    heights = copy(heights)
    heights[col_slice] = calculate_heights(game, col_slice)
    return heights

def update_wells(heights_wells, game, placement, filled_rows):
  heights, wells = heights_wells
  heights = update_heights(heights, game, placement, filled_rows)
  return calculate_wells(game, heights)

def update_holes_plus(holes, game, placement, filled_rows):
  if placement is None:
    return holes
  if filled_rows:
    return calculate_holes_plus(game)
  else:
    col_low, col_high = placement.col_bounds
    holes_slice = slice(col_low * 2, col_high * 2)
    col_slice = slice(col_low, col_high)
    holes = copy(holes)
    holes[holes_slice] = calculate_holes_plus(game, col_slice)
    return holes

update_feature = {
  'holes': update_holes,
  'landing': update_landing,
  'landing_base': update_landing_base,
  'lines': update_lines,
  # 'heights': update_heights,
  # 'wells': update_wells,
  'holes+': update_holes_plus
}


def update(feature, prev_value, game, placement, filled_rows = ()):
  try:
    return update_feature[feature](prev_value, game, placement, filled_rows)
  except KeyError:
    return 0


class Evaluation(object):
  def __init__(self, game, feature_weights):
    self.game = game
    self.weights = {
      f: broadcast(f, self.game, v) for f, v in feature_weights.items()
    }
    self.saved_values = {
      f: [calculate(f, game)] for f in feature_weights if f in update_feature
    }

  def update(self, placement, filled_rows, _):
    for feature, values in self.saved_values.items():
      values.append(update(feature, values[-1], self.game,
                           placement, filled_rows))

  def rollback(self):
    for values in self.saved_values.values():
      values.pop()

  def calculate(self, feature):
    if isinstance(feature, tuple):
      return np.product([self.calculate(x) for x in feature])
    if feature in self.saved_values:
      return self.saved_values[feature][-1]
    return calculate(feature, self.game)

  def value(self):
    score = sum(np.dot(self.calculate(x), self.weights[x]) for x in self.weights)
    if self.game.own.field.ceiling == 0:
      score += -1000 * np.count_nonzero(self.game.own.field.cells[0,3:7])
      score += -100 * np.count_nonzero(self.game.own.field.cells[0,0:3])
      score += -100 * np.count_nonzero(self.game.own.field.cells[0,7:10])
    return score

