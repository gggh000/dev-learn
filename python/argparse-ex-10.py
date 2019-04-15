
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", type=int, choices=[0,1,2,3],
                    help="increase output verbosity")
args = parser.parse_args()
f args.verbosity:
	print(args.verbosity ** 2)
else:
	print("Verbosity is empty.")

	
	
