'''
print('Mandatory argument accepting integers')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number", type=int)
args = parser.parse_args()
print(args.square**2)
print('---------------')

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