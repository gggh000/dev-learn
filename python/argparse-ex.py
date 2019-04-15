'''
print('Mandatory argument accepting integers')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number", type=int)
args = parser.parse_args()
print(args.square**2)
print('---------------')

'''
'''
print('Optional argument that is switched on or off')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", help="increase output verbosity", action="store_true")
args = parser.parse_args()

if args.verbose:
    print("verbosity turned on")
else:
	print("verbosity off")
	
'''

'''
print("combining position and optional argument, no binding")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()
answer = args.square**2

if args.verbose:
    print("the square of {} equals {}".format(args.square, answer))
else:
    print(answer)	
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbosity", type=int,
                    help="increase output verbosity")
args = parser.parse_args()
answer = args.square**2
if args.verbosity == 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity == 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)

	
	
