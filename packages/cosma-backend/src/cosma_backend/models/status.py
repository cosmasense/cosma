import enum


class ProcessingStatus(enum.Enum):
    DISCOVERED = 0
    PARSED = 1
    SUMMARIZED = 2
    COMPLETE = 3

    FAILED = 4
