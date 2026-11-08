"""Evaluate the person-detection benchmark."""

import sys
from benchmark import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
