#!/usr/bin/env python

from __future__ import print_function, division
from collections import Sequence
from planner import SearchPlanner
from game import Game
from multiprocessing import Pool, freeze_support, cpu_count
from copy import copy
from statistics import stdev

import cProfile
import random
import math
import numpy as np
import heapq
import game
import pickle
import time
import itertools

import sys
if sys.version_info[0] < 3:
    range = xrange

from pprint import pprint

mean = lambda x: sum(x) / len(x)

def normalize(feature_weights): # in place
  magnitude = math.sqrt(sum(np.sum(np.dot(l, l)) for l in feature_weights.values()))
  for f, w in feature_weights.items():
    if isinstance(w, tuple):
      feature_weights[f] = tuple(x / magnitude for x in w)
    else:
      feature_weights[f] = w / magnitude
  return feature_weights

def generate(feature_lengths):
  length = sum(l for l in feature_lengths.values())
  weights = (random.gauss(0, 1) for _ in range(length))

  feature_weights = {}
  for f, l in feature_lengths.items():
    feature_weights[f] = ( next(weights) if l == 1 else
                           np.array([next(weights) for _ in range(l)]) )
  return normalize(feature_weights)

def average(fw1, fw2, w1 = 0.5):
  w2 = 1 - w1

  def weighted_sum(f):
    return np.multiply(w1, fw1.get(f, 0)) + np.multiply(w2, fw2.get(f, 0))

  feature_weights = { f: weighted_sum(f) for f in set(fw1) | set(fw2) }
  return normalize(feature_weights)

def mutate(feature_weights, weight = 0.1):
  feature_lengths = { f: len(w) if isinstance(w, Sequence) else 1
                      for f, w in feature_weights.items() }
  mutation_weights = generate(feature_lengths)
  return average(feature_weights, mutation_weights, 1 - weight)

def mutate_axis(feature_weights, feature = None, weight = 0.1):
  if feature is None:
    feature = random.choice(list(feature_weights.keys()))
  mutation_weights = copy(feature_weights)
  mutation_weights[feature] += random.gauss(0, 1) * weight
  return normalize(mutation_weights)

def crossover(fw1, fw2, score1, score2):
  try:
    w1 = score1 / (score1 + score2)
  except ZeroDivisionError:
    w1 = 0.5
  return average(fw1, fw2, w1)

def simulate(feature_weights, pieces, planner_cls = SearchPlanner):
  game = Game(piece = pieces[0], next_piece = pieces[1],
              solid_rate = 15, garbage_rate = 10)

  # game.own.field.cells[19,:9] = True
  # game.own.field.cells[18,2:] = True
  # game.own.field.cells[17,:9] = True
  # game.own.field.cells[16,1:] = True
  # game.own.field.reset_ceiling()

  planner = planner_cls(game, 2, feature_weights) 
  length = 0
  for piece in pieces[2:]:
    move = planner.move()
    if move is None:
      break
    game.own.move(move)
    if not game.advance(piece):
      break
    length += 1
  return game.own.points * 2 + length

def fitness(feature_weights, piece_lists):
  points = sum(simulate(feature_weights, pieces) for pieces in piece_lists)
  avg_points = points / len(piece_lists)
  return avg_points

def tournament(fitnesses, select_prop = 0.125, offspring_pop = 0.25):
  num_select = max(2, int(select_prop * len(fitnesses)))
  num_offspring = max(1, int(offspring_pop * len(fitnesses)))

  def select_parents():
    group = [(x, sum(y) / len(y))
             for x, y in random.sample(fitnesses, num_select)]
    return heapq.nlargest(2, group, key=lambda x: x[1])

  for _ in range(num_offspring):
    (parent1, score1), (parent2, score2) = select_parents()
    yield crossover(parent1, parent2, score1, score2)

def delete_last(fitnesses, delete_prop = 0.25, sort = True):
  num_keep = int((1 - delete_prop) * len(fitnesses))
  if sort:
    fitnesses = sorted(fitnesses,
                       key=lambda x: sum(x[1]) / len(x[1]),
                       reverse=True)
  return fitnesses[:num_keep]

def process(entry, piece_lists, fixed_games = False):
  if not isinstance(entry, tuple):
    entry = (entry, [fitness(entry, piece_lists)])
  elif not fixed_games:
    for piece_list in piece_lists:
      entry[1].append(fitness(entry[0], [piece_list]))
  return entry

def evolve(start_generation, generations = None, num_generations = 50,
           num_pieces = 120, num_games = 25, fixed_games = True,
           mutation_rate = 0.05, dump_file = 'evo.bin', 
           display_progress = True, cpus = 6):
  pool = Pool(processes = cpus)

  if display_progress:
    print('started pool with {} processes'.format(cpus))

  def rand_piece_lists():
    return [[random.choice(game.pieces) for _ in range(num_pieces)]
             for _ in range(num_games)]

  if generations is None:
    generations = []
  curgen = start_generation
  piece_lists = rand_piece_lists()

  for i in (range(num_generations) if num_generations != -1 else itertools.count()):
    if display_progress:
      print('generation {} ({})'.format(i, (time.strftime("%H:%M:%S"))))

    fitnesses = pool.starmap(process, zip(curgen,
                                          itertools.repeat(piece_lists),
                                          itertools.repeat(fixed_games)))
    fitnesses.sort(key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

    generations.append(fitnesses)

    keep = list(x #if fixed_games else x[0]
                for x in delete_last(fitnesses, sort = False))
    new = list(mutate(x) if random.random() < mutation_rate else x
               for x in tournament(fitnesses) )

    curgen = keep + new

    if not fixed_games:
      piece_lists = rand_piece_lists()

    pickle.dump(generations, open(dump_file, 'wb'))

    if display_progress:
      avg_fitness = sum(mean(x[1]) for x in fitnesses) / len(fitnesses)
      range_fitness = mean(fitnesses[-1][1]), 'to', mean(fitnesses[0][1])
      print('fitness average:', avg_fitness, '; range:', range_fitness)
      print('generation best:')
      pprint(fitnesses[0][0])
      print('----')

  if display_progress:
    print('done ({})'.format((time.strftime("%H:%M:%S"))))
    display_summary(generations)

  return generations

def display_summary(generations, top_n = 5):
  overall = {}
  for generation in generations:
    for gene, scores in generation:
      key = frozenset((k, tuple(x for x in v) if isinstance(v, np.ndarray) else v)
                      for k, v in gene.items())
      if key not in overall:
        overall[key] = []
      overall[key] += scores

  avgs = [(dict(k), mean(v), stdev(v), len(v)) for k, v in overall.items()
          if len(v) > 1]
  avgs.sort(key = lambda x: x[1] - (1.96 * x[2] / math.sqrt(x[3])) , reverse = True)

  print('individuals:', len(avgs))

  print('overall best (w/ mean, stdev, and number of scores):')
  pprint(avgs[:5])

def seed(size, feature_lengths, preexisting = []):
  size -= len(preexisting)
  return preexisting + [generate(feature_lengths) for _ in range(size)]

def main():
  start_generation = seed(256,
                          {'ceiling2': 1,
                            'combo': 1,
                            'heights': 1,
                            'holes+': 2,
                            'landing': 1,
                            'lines': 1,
                            'points': 1,
                            'tspin': 1,
                            'wells': 1})
  evolve(start_generation)

if __name__=="__main__":
  freeze_support()
  main()
  # cProfile.run('main()')
