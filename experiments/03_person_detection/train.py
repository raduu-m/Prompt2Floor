"""Train the person-detection benchmark."""

import sys
from benchmark import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
