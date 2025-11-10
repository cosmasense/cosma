from multiprocessing import freeze_support
from backend import serve

if __name__ == "__main__":
    freeze_support()
    serve()
