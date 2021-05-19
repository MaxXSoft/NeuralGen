__version__ = '0.0.1'
__license__ = 'GPLv3'


def main() -> None:
  import argparse
  import json
  from neural_gen.network import from_dict
  from neural_gen.generator import CppGenerator

  # initialize parser
  parser = argparse.ArgumentParser(prog='neural_gen')
  parser.formatter_class = argparse.RawTextHelpFormatter
  parser.description = 'NeuralGen is a naive neural network framework/generator.\n' + \
      'Copyright (C) 2010-2021 MaxXing. License GPLv3.'
  parser.add_argument('descriptor', type=str,
                      help='neural network descriptor (json)')
  parser.add_argument('-g', '--gen', default='cpp', type=str,
                      help='type of generator (cpp), default to "cpp"')
  parser.add_argument('-o', '--output', type=str,
                      help='file name of generated code')
  parser.add_argument('-v', '--version', action='version',
                      version=f'%(prog)s {__version__}')

  # parse arguments
  args = parser.parse_args()
  if not args.descriptor or not args.output:
    parser.print_help()
    exit(1)

  # load network
  with open(args.descriptor, 'r') as f:
    network = from_dict(json.load(f))

  # generate code
  gen = {
      'cpp': CppGenerator,
  }[args.gen]()
  gen.generate(network)
  with open(args.output, 'w') as f:
    gen.dump(f)
