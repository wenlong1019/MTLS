import os


def get_filename(log_dir, load):
    if load is None:
        return os.path.join(log_dir, 'train_log.txt')
    else:
        return os.path.join(log_dir, 'test_log.txt')


class Logger(object):
    def __init__(self, stream):
        self.log = None
        self.terminal = stream

    def set_log(self, log_dir, load):
        filename = get_filename(log_dir, load)
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        pass
