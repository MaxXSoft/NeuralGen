#!/usr/bin/env python3

from os import listdir, path
from sys import argv
from typing import List, Tuple, Any, Iterator
from subprocess import check_output, DEVNULL, CalledProcessError
from resource import getrusage, RUSAGE_CHILDREN


def chunks(l: List[Any], n: int) -> Iterator[List[Any]]:
  for i in range(0, len(l), n):
    yield l[i:i + n]


def check_case(args: List[str], files: List[str]) -> int:
  expected = [i[:-4].split('-')[-1] for i in files]
  out = check_output(args + files, stderr=DEVNULL)
  out = out.decode('utf-8').strip().split('\n')
  correct = 0
  for (x, y) in zip(expected, out):
    if x == y:
      correct += 1
  return correct


def check(args: List[str], test_dir: str) -> Tuple[int, int]:
  total = 0
  correct = 0
  for c in chunks(listdir(test_dir), 50):
    total += len(c)
    correct += check_case(args, list(map(lambda x: path.join(test_dir, x), c)))
  return correct, total


if __name__ == '__main__':
  if len(argv) < 4:
    print(f'Usage: {argv[0]} <ARGS ...> TEST_DIR')
    exit(1)
  try:
    c, t = check(argv[1:-1], argv[-1])
    usage = getrusage(RUSAGE_CHILDREN)
    print(f'Correct/Total: {c}/{t}')
    print(f'Correct Rate: {c * 100 / t}%')
    print(f'Total User Time: {usage.ru_utime * 1000:.2f}ms')
    print(f'Total System Time: {usage.ru_stime * 1000:.2f}ms')
  except CalledProcessError:
    print('Failed to run network!')
    exit(1)
