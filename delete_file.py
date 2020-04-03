import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='+', help="path to files to delete")
    args = parser.parse_args()

    print(f"deleting files and folders...")
    for f_name in args.path:
        if os.path.isfile(f_name):
            os.remove(f_name)
            print(f"file '{f_name}' is deleted")
        elif os.path.isdir(f_name):
            shutil.rmtree(f_name)
            print(f"folder '{f_name}' is deleted")
        else:
            print(f"{f_name} was already removed!")


if __name__ == '__main__':
    main()
