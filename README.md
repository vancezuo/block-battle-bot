# block-battle-bot
This is a python bot that was designed to play [block battle](http://theaigames.com/competitions/ai-block-battle), a multiplayer tetris variant, in a competition at theaigames.com. Its key points:

- Breadth first search move generation
- Beam expectimax search (ignoring opponent moves)
- Evaluation function with the following features:
  - Points
  - Combo counter
  - Last placed piece height
  - Column heights
  - Holes
  - Wells
  - T-spin progress
- Genetic algorithm tuning of feature weights vector, using:
  - Fitness function based on single-player tetris simulations
  - Tournament selection with weighted average crossover
  - Random rotation mutation
- Opening book

[Yiyuan Lee's tetris AI](https://github.com/LeeYiyuan/tetrisai) provided invaluable inspiration for this bot's evaluation features and genetic algorithm design.

The bot is rather slow despite various optimizations with caching and using numpy vector operations, managing at best depth 2 full-width searches in the competition's time controls. That's python for you. Nonetheless, it was able to reach the [competition's round-of-24 semifinals](http://theaigames.com/discussions/ai-block-battle/5743198c5d203c2629cc86c7/win-probability-statistics/1/show) -- a pleasant surprise.

## Requirements
- Python

The bot was designed to run on Python 2 in the competition, but should also be compatible with Python 3.

## Usage
To run the bot in TheAIGames, run `submit.py` and upload the output zip (named "bbai.zip" by default) to the competition platform. It is also possible to run it directly in console via `bot.py`. Refer to the [protocol](http://theaigames.com/competitions/ai-block-battle/getting-started) used in the competition for how to communicate with the bot.

To run a single-player tetris simulation, see `test.py`.

To run genetic algorithm tuning, see `evo.py`.

To generate an opening book, see `gen_openings.py`. Note this repo contains a precomputed book with depth 3 search results for the first three moves of a game.
