
import argparse


parser = argparse.ArgumentParser(description="Test for the argparser")

parser.add_argument("--test_number", 
                     default=30, 
                     type=int, 
                     help="set any integer to use")

args = parser.parse_args()

TEST_NUM = args.test_number

print('Selected number is {}'.format(TEST_NUM))
