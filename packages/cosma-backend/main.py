from multiprocessing import freeze_support
from cosma_backend import serve

if __name__ == "__main__":
    freeze_support()
    serve()
